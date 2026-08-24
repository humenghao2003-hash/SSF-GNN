"""Download, crop, and stack a Sentinel-2 L2A wildfire scene for SSF-GNN.

This uses the public Planetary Computer STAC catalogue and its short-lived
read-only URL signatures.  The output retains the exact 12-band order used by
the project; the scene is cropped to an approximately 15 x 15 km study area.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from osgeo import gdal


ROOT = Path("data/exp")
SOURCE = ROOT / "maui_source_bands"
OUT = ROOT / "maui_wildfire_lahaina_2023-08-18_S2L2A_12band_15km.tif"
ITEMS = Path("/tmp/maui_stac.json")
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
# Native UTM Zone 4N metres: Lahaina and surrounding burn area.
BOUNDS = (730000, 2302000, 745360, 2317360)  # exact 1536 x 1536 at 10 m


def signed_url(url: str) -> str:
    endpoint = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?"
    with urlopen(endpoint + urlencode({"href": url}), timeout=60) as response:
        return json.load(response)["href"]


def selected_item() -> dict:
    items = json.loads(ITEMS.read_text())["features"]
    return next(item for item in items if item["id"].endswith("T04QGJ_20230819T025016"))


def main() -> None:
    if OUT.exists():
        print(f"Already prepared: {OUT}")
        return
    item = selected_item()
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    warped = []
    options = gdal.WarpOptions(
        format="MEM", outputBounds=BOUNDS, xRes=10, yRes=10,
        resampleAlg="bilinear", targetAlignedPixels=True,
    )
    for band in BANDS:
        print(f"Cropping/resampling {band}…", flush=True)
        image = gdal.Warp(f"/vsimem/maui_{band}.tif", "/vsicurl/" + signed_url(item["assets"][band]["href"]), options=options)
        if image is None:
            raise RuntimeError(f"Could not process {band}")
        warped.append(image)
    output = gdal.GetDriverByName("GTiff").Create(
        str(OUT), 1536, 1536, 12, gdal.GDT_UInt16,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    output.SetGeoTransform(warped[0].GetGeoTransform())
    output.SetProjection(warped[0].GetProjection())
    for number, (name, image) in enumerate(zip(BANDS, warped), start=1):
        output.GetRasterBand(number).WriteArray(image.GetRasterBand(1).ReadAsArray())
        output.GetRasterBand(number).SetDescription(name)
    output.FlushCache()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
