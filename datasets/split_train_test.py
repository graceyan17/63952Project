import pandas as pd
import numpy as np

df = pd.read_csv("datasets/joined_complete_info.csv")

df.head(0).to_csv("datasets/train_edges.csv", index=False)
df.head(0).to_csv("datasets/test_edges.csv", index=False)

test_fraction = 0.5

for user, group in df.groupby("User-ID"):
    n = len(group)
    if n <= 1:
        group.to_csv("datasets/train_edges.csv", mode="a", header=False, index=False)
        continue

    group = group.sample(frac=1)
    n = len(group)
    test_size = max(1, int(n * test_fraction))

    test_idx = np.random.choice(group.index, size=test_size, replace=False)
    test_rows = group.loc[test_idx]
    train_rows = group.drop(test_idx)

    train_rows.to_csv("datasets/train_edges.csv", mode="a", header=False, index=False)
    test_rows.to_csv("datasets/test_edges.csv", mode="a", header=False, index=False)

print("Done splitting.")

## SANITY CHECK! 
train = pd.read_csv("datasets/train_edges.csv")
test = pd.read_csv("datasets/test_edges.csv")

overlap = train.merge(test, on=["User-ID", "ISBN"], how="inner")

print("Number of duplicated edges across train/test:", len(overlap))
print(overlap.head())