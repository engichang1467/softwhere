from pathlib import Path
import io

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

SRC = Path("data/imagenet-1k/data")
DST = Path("/data/imagenet-1k")

SPLITS = [
    ("test", "test"),
    ("train", "train"),
    ("validation", "validation"),
]


def to_rgb(value):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def convert_split(parquet_split, out_split):
    files = sorted(SRC.glob(f"{parquet_split}-*.parquet"))

    for shard_idx, path in enumerate(tqdm(files, desc=parquet_split)):
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names
        read_cols = ["image"] + (["label"] if "label" in columns else [])

        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=read_cols)
            rows = table.to_pylist()

            for row_idx, row in enumerate(rows):
                label = row.get("label")
                class_name = f"{int(label):06d}" if label is not None else "unknown"
                out_dir = DST / out_split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)

                image = to_rgb(row["image"])
                filename = f"{out_split}-{shard_idx:05d}-{rg:04d}-{row_idx:06d}.jpg"
                image.save(out_dir / filename, "JPEG", quality=95)


for parquet_split, out_split in SPLITS:
    convert_split(parquet_split, out_split)