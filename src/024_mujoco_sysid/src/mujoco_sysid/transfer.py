import pickle
from pathlib import Path

RESULTS_DIR = Path(__file__).parents[2] / "results"


def main():
    with open(RESULTS_DIR / "results.pkl", "rb") as f:
        r = pickle.load(f)

    print(r)


if __name__ == "__main__":
    main()
