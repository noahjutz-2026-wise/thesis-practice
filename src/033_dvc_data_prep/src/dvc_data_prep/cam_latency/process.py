import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from dvc_data_prep.cam_latency.generate import PQ_PATH

CSV_PATH = Path(__file__).parents[3] / "outs" / "cam_latency.csv"


def main():
    os.makedirs(CSV_PATH.parent, exist_ok=True)
    df = pd.read_parquet(PQ_PATH)
    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
