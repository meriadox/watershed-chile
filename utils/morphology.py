"""
utils/morphology.py
===================
Morphological parameters of a watershed computed from its polygon and a DEM.

Parameters computed
-------------------
Planimetry:
  A_km2   Area (km²)
  P_km    Perimeter (km)
  Kc      Gravelius compactness coefficient = P / (2√(πA))   [-]
  Ff      Form factor = A / L²                               [-]
  Rc      Circularity ratio = 4πA / P²                       [-]
  Re      Elongation ratio = 2√(A/π) / L                     [-]

Elevations (m a.s.l.):
  Hmin    Minimum elevation (outlet)
  Hmax    Maximum elevation
  Hmed    Mean elevation (pixel average)
  H       Total relief = Hmax − Hmin  (m)
  Hm      Effective relief = Hmed − Hmin  (m)

Drainage:
  L_km    Main channel length (km) — D8 trace from outlet along max-accumulation path
  Lcu_km  Maximum flow path length (km)
  Scp     Main channel slope (m/m)
  Scu     Mean basin slope (m/m)
  Dd      Drainage density (km/km²)
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pysheds.grid import Grid as PyshedsGrid
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

# For upstream tracing: D8 code a neighbour (dr, dc) must have to flow INTO (r, c)
_D8_FROM = {
    64:  (+1,  0),
    128: (+1, -1),
    1:   ( 0, -1),
    2:   (-1, -1),
    4:   (-1,  0),
    8:   (-1, +1),
    16:  ( 0, +1),
    32:  (+1, +1),
}
_D8_DIAGONAL = (128, 2, 8, 32)


def calculate_morphology(
    watershed_shp: Union[str, Path],
    dem_path: Union[str, Path],
    acc_threshold: int = 500,
    crs_work: str = "EPSG:32719",
) -> pd.Series:
    """
    Calculate morphological parameters of a watershed.

    Parameters
    ----------
    watershed_shp : str or Path
        Watershed boundary shapefile (output of delineate_watershed()).
    dem_path : str or Path
        GeoTIFF DEM covering the watershed (output of download_dem()).
    acc_threshold : int
        Flow accumulation threshold (cells) used to define the stream network.
        Should match the value used in delineate_watershed().
    crs_work : str
        Projected CRS (metres) for area and perimeter calculations.
        Default: EPSG:32719 (UTM 19S, central Chile).

    Returns
    -------
    pd.Series
        All morphological parameters as a named Series.
    """
    watershed_shp = Path(watershed_shp)
    dem_path = Path(dem_path)

    # -- Planimetry from the projected polygon --
    gdf_proj = gpd.read_file(watershed_shp).to_crs(crs_work)
    ws_proj = (
        gdf_proj.union_all()
        if hasattr(gdf_proj, "union_all")
        else gdf_proj.unary_union
    )
    A_km2 = ws_proj.area / 1e6
    P_km  = ws_proj.length / 1000.0

    # -- DEM metadata --
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        orig_transform = src.transform
        cellsize_m = _cellsize_m(src)

    # -- Reproject watershed to DEM CRS for masking --
    gdf_dem = gpd.read_file(watershed_shp).to_crs(dem_crs)
    ws_dem = (
        gdf_dem.union_all()
        if hasattr(gdf_dem, "union_all")
        else gdf_dem.unary_union
    )
    ws_dem_buf = ws_dem.buffer(5 * abs(orig_transform.a))

    # -- Run D8 pipeline in a temp directory --
    with tempfile.TemporaryDirectory() as td:
        dem_clip_path = Path(td) / "dem_clip.tif"
        _NODATA = -9999.0

        img, tr, meta = _clip_dem(dem_path, ws_dem_buf, nodata=_NODATA)
        with rasterio.open(dem_clip_path, "w", **meta) as dst:
            dst.write(img)

        grid = PyshedsGrid.from_raster(str(dem_clip_path))
        dem_arr = grid.read_raster(str(dem_clip_path))
        fdir = grid.flowdir(grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem_arr))))
        acc  = grid.accumulation(fdir)

        catch_mask = rasterize(
            [(mapping(ws_dem), 1)],
            out_shape=(meta["height"], meta["width"]),
            transform=tr,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        flooded_arr = np.asarray(grid.fill_depressions(grid.fill_pits(dem_arr))).astype(np.float64)
        acc_arr     = np.asarray(acc)
        fdir_arr    = np.asarray(fdir).astype(np.int32)

        # Outlet: cell with maximum accumulation inside the catchment
        acc_in_catch = np.where(catch_mask, acc_arr, 0)
        r_out, c_out = np.unravel_index(np.argmax(acc_in_catch), acc_in_catch.shape)
        aff = grid.affine
        outlet_x = aff.c + (c_out + 0.5) * aff.a
        outlet_y = aff.f + (r_out + 0.5) * aff.e

        # Elevations
        dem_in_ws = np.where(catch_mask, flooded_arr, np.nan)
        Hmin = float(flooded_arr[r_out, c_out])
        Hmax = float(np.nanmax(dem_in_ws))
        Hmed = float(np.nanmean(dem_in_ws))
        H  = Hmax - Hmin
        Hm = Hmed - Hmin

        # Mean basin slope (gradient of filled DEM)
        grad_y, grad_x = np.gradient(np.where(catch_mask, flooded_arr, np.nan), cellsize_m)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        Scu = float(np.nanmean(slope[catch_mask]))

        # Main channel: D8 upstream trace following maximum accumulation
        rows_g, cols_g = fdir_arr.shape
        r, c = r_out, c_out
        r_head, c_head = r, c
        L_m = 0.0
        visited: set = {(r, c)}

        while True:
            best_acc, best_pos = -1, None
            for code, (dr, dc) in _D8_FROM.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows_g and 0 <= nc < cols_g:
                    if (
                        fdir_arr[nr, nc] == code
                        and acc_arr[nr, nc] > acc_threshold
                        and catch_mask[nr, nc]
                        and (nr, nc) not in visited
                    ):
                        if acc_arr[nr, nc] > best_acc:
                            best_acc, best_pos = acc_arr[nr, nc], (nr, nc)
            if best_pos is None:
                break
            nr, nc = best_pos
            diagonal = abs(nr - r) == 1 and abs(nc - c) == 1
            L_m += cellsize_m * (math.sqrt(2) if diagonal else 1.0)
            r_head, c_head = nr, nc
            r, c = nr, nc
            visited.add((r, c))

        L_km   = max(L_m / 1000.0, cellsize_m / 1000.0)
        Hmax_cp = float(flooded_arr[r_head, c_head])
        Scp    = (Hmax_cp - Hmin) / L_m if L_m > 0 else 0.0

        # Maximum flow-path length (distance_to_outlet with diagonal weights)
        is_diag = np.isin(fdir_arr, _D8_DIAGONAL)
        weights = np.where(is_diag, cellsize_m * math.sqrt(2), cellsize_m).astype(np.float64)
        try:
            dist = grid.distance_to_outlet(x=outlet_x, y=outlet_y, fdir=fdir,
                                           weights=weights, xytype="coordinate")
            dist_arr = np.asarray(dist, dtype=np.float64)
            dist_arr = np.where(np.isfinite(dist_arr), dist_arr, 0.0)
            if dist_arr.max() < cellsize_m:
                raise RuntimeError("distance_to_outlet returned zeros")
        except Exception:
            try:
                dist = grid.distance_to_outlet(x=outlet_x, y=outlet_y,
                                               fdir=fdir, xytype="coordinate")
                dist_arr = np.asarray(dist, dtype=np.float64) * cellsize_m
                dist_arr = np.where(np.isfinite(dist_arr), dist_arr, 0.0)
            except Exception:
                dist_arr = np.zeros(fdir_arr.shape, dtype=np.float64)

        dist_catch = np.where(catch_mask & (dist_arr > 0), dist_arr, 0.0)
        Lcu_km = max(float(np.max(dist_catch)) / 1000.0, L_km)

        # Drainage density
        n_stream = int(np.sum((acc_arr > acc_threshold) & catch_mask))
        Dd = (n_stream * cellsize_m / 1000.0) / A_km2 if A_km2 > 0 else 0.0

    # Shape indices (with L in km, A in km²)
    Kc = P_km / (2.0 * math.sqrt(math.pi * A_km2)) if A_km2 > 0 else np.nan
    Ff = A_km2 / (L_km**2)                          if L_km  > 0 else np.nan
    Rc = (4.0 * math.pi * A_km2) / (P_km**2)        if P_km  > 0 else np.nan
    Re = (2.0 / L_km) * math.sqrt(A_km2 / math.pi)  if L_km  > 0 else np.nan

    return pd.Series({
        "A_km2":  round(A_km2,  2),
        "P_km":   round(P_km,   2),
        "Kc":     round(Kc,     3),
        "Ff":     round(Ff,     4),
        "Rc":     round(Rc,     3),
        "Re":     round(Re,     3),
        "Hmin":   round(Hmin,   1),
        "Hmax":   round(Hmax,   1),
        "Hmed":   round(Hmed,   1),
        "H":      round(H,      1),
        "Hm":     round(Hm,     1),
        "L_km":   round(L_km,   2),
        "Lcu_km": round(Lcu_km, 2),
        "Scp":    round(Scp,    5),
        "Scu":    round(Scu,    5),
        "Dd":     round(Dd,     3),
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cellsize_m(src: rasterio.DatasetReader) -> float:
    if src.crs.is_projected:
        return abs(src.transform.a)
    lat = (src.bounds.top + src.bounds.bottom) / 2.0
    return abs(src.transform.a) * 111_000.0 * np.cos(np.radians(abs(lat)))


def _clip_dem(dem_path, geometry, nodata: float = -9999.0, buf_cells: int = 5):
    """Clip DEM to geometry bounds. Uses windowed read for large files (>500 MB)."""
    from rasterio.windows import from_bounds as win_from_bounds

    large = Path(dem_path).stat().st_size > 500 * 1024 * 1024

    with rasterio.open(dem_path) as src:
        if large:
            minx, miny, maxx, maxy = geometry.bounds
            res = abs(src.transform.a)
            win = win_from_bounds(
                minx - res * buf_cells, miny - res * buf_cells,
                maxx + res * buf_cells, maxy + res * buf_cells,
                src.transform,
            )
            win = win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            img = src.read(1, window=win)
            tr  = src.window_transform(win)
            meta = src.meta.copy()
            meta.update({"height": img.shape[0], "width": img.shape[1], "transform": tr})
            img = img.astype(np.float32)
            if src.nodata is not None:
                img[img == float(src.nodata)] = nodata
            img[np.isnan(img)] = nodata
            geom_mask = rasterize(
                [(mapping(geometry), 1)],
                out_shape=img.shape, transform=tr, fill=0, dtype="uint8",
            )
            img[geom_mask == 0] = nodata
            meta.update({"dtype": "float32", "nodata": nodata, "count": 1})
            return img[np.newaxis, :, :], tr, meta
        else:
            img, tr = rio_mask(src, [mapping(geometry)], crop=True)
            meta = src.meta.copy()
            img = img.astype(np.float32)
            if src.nodata is not None:
                img[img == float(src.nodata)] = nodata
            img[np.isnan(img)] = nodata
            meta.update({
                "height": img.shape[1], "width": img.shape[2],
                "transform": tr, "dtype": "float32", "nodata": nodata,
            })
            return img, tr, meta
