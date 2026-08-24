"""Create a compact JPEG preview from the downloaded Sentinel-2 true-colour TIFF."""
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image


source = Path("data/exp/maui_wildfire_lahaina_2023-08-18_preview.tif")
target = source.with_suffix(".jpg")
dataset = gdal.Open(str(source))
thumbnail = dataset.ReadAsArray(
    buf_xsize=2048, buf_ysize=2048,
).transpose(1, 2, 0)
# Sentinel-2 TCI is already RGB. A percentile stretch makes the landscape
# legible while avoiding a few very bright clouds dominating the image.
low, high = np.percentile(thumbnail, (1, 99))
preview = np.clip((thumbnail - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
Image.fromarray(preview).save(target, quality=92)
print(target)
