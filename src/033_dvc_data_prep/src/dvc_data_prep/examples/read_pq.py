import pandas as pd
from pyarrow import parquet as pq

from dvc_data_prep.processors.cam_latency import PQ_PATH


def main():
    print("using pandas")
    df = pd.read_parquet(PQ_PATH)
    print(df)

    print("cvt to numpy")
    ndarr = df.to_numpy()
    print(ndarr)

    print("using pyarrow.parquet")
    tbl = pq.read_table(PQ_PATH)
    print(tbl)


if __name__ == "__main__":
    main()
