"""
utils/delineation.py
====================
Watershed delineation using pysheds and Copernicus GLO-30 DEM.

Main workflow:
  1. download_dem(bbox)             — download Copernicus tiles (free, no credentials)
  2. delineate_watershed(lon, lat)  — D8 delineation + stream network extraction
  3. snap_to_network(lon, lat, shp) — optional: snap outlet to official network (BCN, etc.)

Compatibility: pysheds 0.3.5 requires numpy < 2.0
  If you see "module 'numpy' has no attribute 'bool8'", apply the patch:
    sed -i 's/np\.bool8/np.bool_/g' $(python -c "import pysheds; print(pysheds.__file__.replace('__init__.py',''))")/{sgrid,sview,_sgrid}.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
from pysheds.grid import Grid as PyshedsGrid
from rasterio.features import shapes
from rasterio.merge import merge as rio_merge
from shapely.geometry import LineString, shape
from shapely.ops import linemerge

logger = logging.getLogger(__name__)

_COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# D8 direction code → (delta_row, delta_col) of downstream neighbour
_D8_TO = {
    64: (-1, 0), 128: (-1, 1), 1: (0, 1),  2: (1, 1),
     4: ( 1, 0),   8: ( 1,-1), 16: (0,-1), 32: (-1,-1),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_dem(bbox, cache_dir="cache"):
    """
    Download Copernicus GLO-30 DEM tiles covering a bounding box.

    No credentials required — data is served openly from AWS S3.

    Parameters
    ----------
    bbox : list or tuple
        [xmin, ymin, xmax, ymax] in WGS84 decimal degrees.
    cache_dir : str or Path
        Directory to cache downloaded tiles. Tiles are reused on subsequent
        calls, so the download only happens once per tile.

    Returns
    -------
    Path
        Path to the downloaded GeoTIFF. If more than one tile is needed,
        a mosaicked file is returned.

    Example
    -------
    >>> bbox = [-71.0, -28.5, -68.5, -26.0]  # Copiapó basin
    >>> dem_path = download_dem(bbox, cache_dir="cache")
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = bbox
    margin = 0.05  # ~5.5 km extra on each side

    lat_min = int(np.floor(ymin - margin))
    lat_max = int(np.floor(ymax + margin))
    lon_min = int(np.floor(xmin - margin))
    lon_max = int(np.floor(xmax + margin))

    tiles = []
    for tile_lat in range(lat_min, lat_max + 1):
        for tile_lon in range(lon_min, lon_max + 1):
            ns = "N" if tile_lat >= 0 else "S"
            ew = "E" if tile_lon >= 0 else "W"
            name = (
                f"Copernicus_DSM_COG_10_{ns}{abs(tile_lat):02d}_00"
                f"_{ew}{abs(tile_lon):03d}_00_DEM"
            )
            url = f"{_COP30_BASE}/{name}/{name}.tif"
            fname = f"cop30_{ns}{abs(tile_lat):02d}_{ew}{abs(tile_lon):03d}.tif"
            local = cache_dir / fname

            if local.exists():
                logger.info(f"Using cached tile: {fname}")
            else:
                logger.info(f"Downloading {fname} ...")
                resp = requests.get(url, timeout=300, stream=True)
                if resp.status_code == 404:
                    logger.info(f"{fname}: no data (ocean or outside coverage)")
                    continue
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                _CHUNK = 1 << 20  # 1 MB
                downloaded = 0
                with open(local, "wb") as fh:
                    if _HAS_TQDM and total:
                        with _tqdm(total=total, unit="B", unit_scale=True,
                                   desc=fname, leave=False) as bar:
                            for chunk in resp.iter_content(_CHUNK):
                                fh.write(chunk)
                                bar.update(len(chunk))
                                downloaded += len(chunk)
                    else:
                        for chunk in resp.iter_content(_CHUNK):
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                logger.info(f"  {fname}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB  ({pct:.0f}%)")
                logger.info(f"{fname}: {downloaded/1e6:.1f} MB — OK")

            tiles.append(local)

    if not tiles:
        raise FileNotFoundError(
            "No Copernicus tiles found for the given bbox. "
            "Check that coordinates are in WGS84 and the area has land."
        )

    if len(tiles) == 1:
        return tiles[0]

    logger.info(f"Mosaicking {len(tiles)} tiles ...")
    datasets = [rasterio.open(t) for t in tiles]
    mosaic, transform = rio_merge(datasets)
    meta = datasets[0].meta.copy()
    for ds in datasets:
        ds.close()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
    mosaic_path = cache_dir / "dem_mosaic.tif"
    with rasterio.open(mosaic_path, "w", **meta) as dst:
        dst.write(mosaic)
    logger.info(f"Mosaic saved: {mosaic_path.name}")
    return mosaic_path


def delineate_watershed(
    outlet_lon: float,
    outlet_lat: float,
    dem_path,
    out_dir="results",
    acc_threshold: int = 500,
    snap_dist_m: float = 300.0,
    network_threshold: int | None = None,
    crs_output: str = "EPSG:32719",
    extract_network: bool = True,
) -> dict:
    """
    Delineate a watershed from an outlet point using D8 flow routing.

    Parameters
    ----------
    outlet_lon, outlet_lat : float
        Outlet coordinates in WGS84 decimal degrees.
    dem_path : str or Path
        GeoTIFF DEM, e.g. the output of download_dem().
    out_dir : str or Path
        Directory where shapefiles are saved.
    acc_threshold : int
        Flow accumulation threshold (cells) for stream definition.
        At 30 m resolution, 500 cells ≈ 0.45 km² of contributing area.
        Typical range: 200–2000.
    snap_dist_m : float
        Maximum distance (m) to snap the outlet to the nearest stream cell.
        A warning is logged if the actual snap distance exceeds this value,
        but delineation proceeds anyway.
    network_threshold : int, optional
        Accumulation threshold for the exported stream network shapefile.
        Defaults to acc_threshold. Use a larger value (e.g. 10 × acc_threshold)
        to export only the main channels for map display.
    crs_output : str
        Output CRS for all shapefiles. Default: EPSG:32719 (UTM 19S, Chile).
        Change to the appropriate UTM zone for your study area.
    extract_network : bool
        If True, also save the stream network as network.shp.

    Returns
    -------
    dict with keys:
        "watershed"     : GeoDataFrame with the watershed boundary polygon
        "network"       : GeoDataFrame with stream lines (None if not extracted)
        "watershed_shp" : str path to watershed.shp
        "network_shp"   : str path to network.shp (None if not extracted)
    """
    dem_path = Path(dem_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if network_threshold is None:
        network_threshold = acc_threshold

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        src_nodata = src.nodata
        cellsize_m = _cellsize_m(src)

    logger.info(f"DEM: {dem_path.name} | {cellsize_m:.1f} m/px | CRS: {dem_crs.to_epsg()}")

    # Ensure nodata=-9999 for pysheds (it is sensitive to nodata values)
    dem_proc = str(dem_path)
    if src_nodata != -9999.0:
        dem_proc = str(out_dir / "_dem_proc.tif")
        with rasterio.open(dem_path) as src:
            data = src.read(1).astype(np.float32)
            meta = src.meta.copy()
            if src_nodata is not None:
                data[data == float(src_nodata)] = -9999.0
            data[np.isnan(data)] = -9999.0
            meta.update({"dtype": "float32", "nodata": -9999.0})
        with rasterio.open(dem_proc, "w", **meta) as dst:
            dst.write(data, 1)

    # D8 hydrological pipeline
    logger.info("Conditioning DEM: fill_pits → fill_depressions → resolve_flats ...")
    grid = PyshedsGrid.from_raster(dem_proc)
    dem_arr = grid.read_raster(dem_proc)
    pit_filled = grid.fill_pits(dem_arr)
    flooded   = grid.fill_depressions(pit_filled)
    inflated  = grid.resolve_flats(flooded)

    logger.info("Computing D8 flow direction and accumulation ...")
    fdir = grid.flowdir(inflated)
    acc  = grid.accumulation(fdir)

    n_stream = int(np.sum(np.asarray(acc) > acc_threshold))
    logger.info(f"Stream cells (acc > {acc_threshold}): {n_stream:,}")
    if n_stream == 0:
        raise RuntimeError(
            f"No stream cells found with acc_threshold={acc_threshold}. "
            "Reduce acc_threshold or check that the bbox covers the full upstream area."
        )

    # Reproject outlet to DEM CRS
    pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([outlet_lon], [outlet_lat]),
        crs="EPSG:4326",
    ).to_crs(dem_crs)
    x, y = float(pt.geometry.x.iloc[0]), float(pt.geometry.y.iloc[0])

    # Two-level snap strategy:
    #   1st: snap to major streams (acc > 20 × threshold) to avoid minor drains
    #   2nd: fallback to any stream cell
    # Keep Raster type — snap_to_mask requires a pysheds Raster, not a numpy array
    _MAJOR = 20
    streams_major = acc > (acc_threshold * _MAJOR)
    n_major = int(np.sum(np.asarray(streams_major)))
    mask_snap = streams_major if n_major > 0 else (acc > acc_threshold)

    xy_snap = grid.snap_to_mask(mask_snap, np.array([[x, y]]))
    x_s, y_s = float(xy_snap[0, 0]), float(xy_snap[0, 1])

    dist_m = _to_meters(np.sqrt((x - x_s)**2 + (y - y_s)**2), dem_crs, x, y)
    if dist_m > snap_dist_m:
        logger.warning(
            f"Nearest stream at {dist_m:.0f} m "
            f"(snap_dist_m={snap_dist_m:.0f} m). "
            "Consider moving the outlet point closer to the channel."
        )
    else:
        logger.info(f"Outlet snapped {dist_m:.0f} m to {'major ' if n_major > 0 else ''}stream.")

    # Catchment delineation
    logger.info("Delineating catchment ...")
    catch = grid.catchment(x=x_s, y=y_s, fdir=fdir, xytype="coordinate")
    catch_arr = np.asarray(catch).astype(np.uint8)

    polys = [
        shape(geom)
        for geom, _ in shapes(catch_arr, mask=catch_arr, transform=grid.affine)
    ]
    if not polys:
        raise RuntimeError(
            "Catchment delineation returned no polygon. "
            "Check outlet coordinates and bbox coverage."
        )

    watershed_geom = max(polys, key=lambda p: p.area)
    gdf_ws = gpd.GeoDataFrame(geometry=[watershed_geom], crs=dem_crs).to_crs(crs_output)
    area_km2 = gdf_ws.geometry.area.iloc[0] / 1e6
    logger.info(f"Watershed area: {area_km2:,.1f} km²")

    ws_shp = str(out_dir / "watershed.shp")
    gdf_ws.to_file(ws_shp)
    logger.info(f"Saved: watershed.shp")

    # Stream network
    gdf_net, net_shp = None, None
    if extract_network:
        acc_arr  = np.asarray(acc)
        fdir_arr = np.asarray(fdir).astype(np.int32)
        lines = _extract_network(acc_arr, fdir_arr, catch_arr == 1, network_threshold, grid.affine)
        if lines:
            gdf_net = gpd.GeoDataFrame(geometry=lines, crs=dem_crs).to_crs(crs_output)
            net_shp = str(out_dir / "network.shp")
            gdf_net.to_file(net_shp)
            logger.info(f"Saved: network.shp ({len(lines)} line features)")

    return {
        "watershed":     gdf_ws,
        "network":       gdf_net,
        "watershed_shp": ws_shp,
        "network_shp":   net_shp,
    }


def snap_to_network(
    outlet_lon: float,
    outlet_lat: float,
    network_shp,
    max_dist_m: float = 500.0,
) -> tuple[float, float]:
    """
    Snap an outlet point to the nearest line in an official hydrographic network.

    Use this as an optional pre-processing step before delineate_watershed() when
    you have access to an authoritative river network (e.g., BCN Chile, available
    from https://www.bcn.cl/siit/mapoteca — download and clip to your study area
    before loading, as the full national file is very large).

    Parameters
    ----------
    outlet_lon, outlet_lat : float
        Outlet coordinates in WGS84.
    network_shp : str or Path
        Path to the hydrographic network shapefile. Pre-clip to your region
        (e.g., with QGIS or geopandas) before passing to this function.
    max_dist_m : float
        Maximum snap distance in meters. If no line is found within this
        distance, the original coordinates are returned unchanged.

    Returns
    -------
    (float, float)
        Snapped (longitude, latitude) in WGS84, or original if snap fails.

    Example
    -------
    >>> # Clip BCN network to your bbox first, then:
    >>> lon, lat = snap_to_network(-70.335, -27.366, "data/bcn_copiapo.shp")
    >>> result = delineate_watershed(lon, lat, dem_path)
    """
    from shapely.ops import nearest_points

    gdf_net = gpd.read_file(network_shp).to_crs("EPSG:32719")

    pt_utm = (
        gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([outlet_lon], [outlet_lat]),
            crs="EPSG:4326",
        )
        .to_crs("EPSG:32719")
        .geometry.iloc[0]
    )

    union = gdf_net.union_all() if hasattr(gdf_net, "union_all") else gdf_net.unary_union
    pt_snap_utm, _ = nearest_points(pt_utm, union)
    dist_m = pt_utm.distance(pt_snap_utm)

    if dist_m > max_dist_m:
        logger.warning(
            f"No network line within {max_dist_m:.0f} m "
            f"(nearest: {dist_m:.0f} m). Using original outlet."
        )
        return outlet_lon, outlet_lat

    logger.info(f"Snapped {dist_m:.0f} m to hydrographic network.")
    pt_wgs = gpd.GeoDataFrame(geometry=[pt_snap_utm], crs="EPSG:32719").to_crs("EPSG:4326")
    return float(pt_wgs.geometry.x.iloc[0]), float(pt_wgs.geometry.y.iloc[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cellsize_m(src: rasterio.DatasetReader) -> float:
    if src.crs.is_projected:
        return abs(src.transform.a)
    lat = (src.bounds.top + src.bounds.bottom) / 2.0
    return abs(src.transform.a) * 111_000.0 * np.cos(np.radians(abs(lat)))


def _to_meters(dist, crs: rasterio.crs.CRS, x: float, y: float) -> float:
    if crs.is_projected:
        return dist
    return dist * 111_000.0 * np.cos(np.radians(abs(y)))


def _extract_network(acc_arr, fdir_arr, catch_mask, threshold, affine) -> list:
    stream_mask = (acc_arr > threshold) & catch_mask
    rows, cols = np.where(stream_mask)
    segments = []
    for r, c in zip(rows, cols):
        code = int(fdir_arr[r, c])
        if code not in _D8_TO:
            continue
        dr, dc = _D8_TO[code]
        nr, nc = r + dr, c + dc
        if 0 <= nr < stream_mask.shape[0] and 0 <= nc < stream_mask.shape[1]:
            if stream_mask[nr, nc]:
                x1 = affine.c + (c  + 0.5) * affine.a
                y1 = affine.f + (r  + 0.5) * affine.e
                x2 = affine.c + (nc + 0.5) * affine.a
                y2 = affine.f + (nr + 0.5) * affine.e
                segments.append(LineString([(x1, y1), (x2, y2)]))
    if not segments:
        return []
    merged = linemerge(segments)
    if merged.geom_type == "LineString":
        return [merged]
    if merged.geom_type == "MultiLineString":
        return list(merged.geoms)
    return segments
