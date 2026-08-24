"""Download a review-only Sentinel-2 L2A mosaic for the August 2023 Maui wildfire.

The result is a 12-band GeoTIFF at 10 m resolution, covering the Lahaina burn
area and its surroundings.  It is intentionally not tiled or passed to the
classifier; it is for human review before the next step.
"""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

from osgeo import gdal


OUT = Path("data/exp/maui_wildfire_lahaina_2023-08-18_S2L2A_10m.tif")
# west, south, east, north: ~17 x 17 km around the Lahaina / Kaanapali burn scar
BOUNDS_LL = (-156.72, 20.82, -156.56, 20.97)
BASE = (
    "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/04/Q/GJ/2023/08/18/"
    "S2B_MSIL2A_20230818T210929_N0509_R057_T04QGJ_20230819T025016.SAFE/"
    "GRANULE/L2A_T04QGJ_A033688_20230818T210926/IMG_DATA"
)
# The model expects precisely this Sentinel-2 band order.
BANDS = [
    ("B01", "R60m"), ("B02", "R10m"), ("B03", "R10m"), ("B04", "R10m"),
    ("B05", "R20m"), ("B06", "R20m"), ("B07", "R20m"), ("B08", "R10m"),
    ("B8A", "R20m"), ("B09", "R60m"), ("B11", "R20m"), ("B12", "R20m"),
]


def source_url(band: str, resolution: str) -> str:
    filename = f"T04QGJ_20230818T210929_{band}_{resolution[1:]}.tif"
    url = f"{BASE}/{resolution}/{filename}"
    signed = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?" + urlencode({"href": url})
    with urlopen(signed, timeout=30) as response:
        return "/vsicurl/" + json.load(response)["href"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Already present: {OUT}")
        return
    import math
    def utm(lon, lat):
        a,e2,k=6378137.0,0.00669437999014,0.9996; z=int((lon+180)/6)+1
        p,l=math.radians(lat),math.radians(lon); l0=math.radians((z-1)*6-177)
        n=a/math.sqrt(1-e2*math.sin(p)**2); t=math.tan(p)**2; c=e2/(1-e2)*math.cos(p)**2; aa=math.cos(p)*(l-l0)
        m=a*((1-e2/4-3*e2**2/64-5*e2**3/256)*p-(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*p)+(15*e2**2/256+45*e2**3/1024)*math.sin(4*p)-(35*e2**3/3072)*math.sin(6*p))
        return k*n*(aa+(1-t+c)*aa**3/6+(5-18*t+t*t+72*c-58*e2/(1-e2))*aa**5/120)+500000, k*(m+n*math.tan(p)*(aa**2/2+(5-t+9*c+4*c*c)*aa**4/24+(61-58*t+t*t+600*c-330*e2/(1-e2))*aa**6/720))
    west,south=utm(BOUNDS_LL[0],BOUNDS_LL[1]); east,north=utm(BOUNDS_LL[2],BOUNDS_LL[3])
    warped = []
    options = gdal.WarpOptions(
        format="MEM", outputBounds=(west,south,east,north), xRes=10, yRes=10,
        resampleAlg="bilinear", targetAlignedPixels=True,
    )
    for band, resolution in BANDS:
        print(f"Reading {band} ({resolution})")
        raster = gdal.Warp(f"/vsimem/{band}.tif", source_url(band, resolution), options=options)
        if raster is None:
            raise RuntimeError(f"Unable to read {band}")
        warped.append(raster)
    driver = gdal.GetDriverByName("GTiff")
    result = driver.Create(
        str(OUT), warped[0].RasterXSize, warped[0].RasterYSize, len(warped),
        gdal.GDT_UInt16, options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    result.SetGeoTransform(warped[0].GetGeoTransform())
    result.SetProjection(warped[0].GetProjection())
    for index, raster in enumerate(warped, start=1):
        result.GetRasterBand(index).WriteArray(raster.GetRasterBand(1).ReadAsArray())
        result.GetRasterBand(index).SetDescription(BANDS[index - 1][0])
    result.FlushCache()
    print(f"Wrote {OUT} ({result.RasterXSize} x {result.RasterYSize}, 12 bands)")


if __name__ == "__main__":
    main()
