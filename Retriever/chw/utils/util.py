from contextlib import contextmanager
import time
import pandas as pd


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def MRR(results: pd.DataFrame):

    sum = 0

    for idx, (index, row) in enumerate(results.iterrows()):
        for context_idx, context in enumerate(row["context"]):
            find_idx = context.find(row["original_context"])
            if find_idx != -1:
                sum += 1 / (context_idx + 1)
                break

    return sum / len(results)


def Recall(results: pd.DataFrame):
    sum = 0
    for idx, (index, row) in enumerate(results.iterrows()):
        for context in row["context"]:
            find_idx = context.find(row["original_context"])
            if find_idx != -1:
                sum += 1
                break

    return sum / len(results)
