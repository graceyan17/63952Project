import pandas as pd

# df = pd.read_csv("datasets/test_edges.csv")

def compute_recall(test_df, user_id, recommendations):
    true_items = set(
        test_df[test_df["User-ID"] == user_id]["join_title"].unique()
    )

    if len(true_items) == 0:
        return 0

    # Count hits
    hits = len(set(recommendations) & true_items)

    return hits / len(true_items)