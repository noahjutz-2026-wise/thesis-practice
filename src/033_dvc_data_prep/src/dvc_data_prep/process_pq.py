from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from dvc_data_prep.record_pq import PQ_PATH

CSV_PATH = Path(__file__).parents[2] / "out.csv"


def main():
    df = pd.read_parquet(PQ_PATH)
    df.to_csv(CSV_PATH)
