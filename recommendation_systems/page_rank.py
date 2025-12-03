import os
import pandas as pd
import networkx as nx
from recall_at_k import compute_recall

# ==========================
#  Load ratings
# ==========================
# Must have: User-ID, ISBN, Book-Rating
df = pd.read_csv("datasets/joined_complete_info.csv")  # use your merged dataset
edges = pd.read_csv("datasets/train_edges.csv")

df["User-ID"] = df["User-ID"].astype(str)
edges["User-ID"] = edges["User-ID"].astype(str)

# User-item bipartite graph
G = nx.Graph()

# Add user nodes and book nodes with type label
all_users = df["User-ID"].unique()
books = df["join_title"].unique()

G.add_nodes_from(all_users, bipartite="user")
G.add_nodes_from(books, bipartite="item")

# Add edges user <-> book with weight = rating
for _, row in edges.iterrows():
    G.add_edge(row["User-ID"], row["join_title"], weight=int(row["Book-Rating"]))

print("Graph built:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

#  Personalized PageRank
def recommend_for_user(users, user_id, top_k, alpha=0.85):
    """
    alpha: probability of continuing walk
    """
    if user_id not in G:
        raise ValueError("User not in graph")

    # Personalized restart vector
    personalization = {node: 0 for node in G.nodes()}
    personalization[user_id] = 1

    # Compute personalized PageRank
    pr = nx.pagerank(
        G,
        alpha= alpha,
        personalization=personalization,
        weight="weight"
    )

    # Convert dict → sorted list
    pr_series = pd.Series(pr)

    # Only keep book nodes
    user_books = edges[edges["User-ID"] == user_id]["join_title"].unique()
    pr_books = pr_series.drop(list(users))  # drop user nodes

    # Remove books the user already rated
    pr_books = pr_books.drop(user_books, errors="ignore")

    # Top recommendations
    return pr_books.sort_values(ascending=False).head(top_k).index.tolist()


test_df = pd.read_csv("datasets/test_edges.csv")
test_df["User-ID"] = test_df["User-ID"].astype(str)

output_path = "recommendation_systems/page_rank.csv"
if not os.path.exists(output_path):
    pd.DataFrame(columns=["User-ID", "Recs"]).to_csv(output_path, index=False)


recs_df = pd.DataFrame(columns=["User-ID", "Recs"])
users = test_df["User-ID"].unique()
recall = 0

for i, user in enumerate(users):
    recs = recommend_for_user(all_users, user, 50)
    recall += compute_recall(test_df, user, recs)

    # print(f"User {user}, cumulative recall: {recall}")

    # Append this row directly to the CSV
    pd.DataFrame([[user, recs]], columns=["User-ID", "Recs"]).to_csv(
        output_path,
        mode="a",
        header=False,
        index=False
    )

print(recall / len(users))