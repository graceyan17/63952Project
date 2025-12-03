import os
import pandas as pd
import networkx as nx
from recall_at_k import compute_recall
import random

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
users = df["User-ID"].unique()
books = df["join_title"].unique()

G.add_nodes_from(users, bipartite="user")
G.add_nodes_from(books, bipartite="item")

# Add edges user <-> book with weight = rating
for _, row in edges.iterrows():
    G.add_edge(row["User-ID"], row["join_title"], weight=int(row["Book-Rating"]))

to_remove = []
for n in G.nodes():
    wsum = sum(G[n][nbr].get("weight", 1) for nbr in G[n])
    if wsum == 0:
        to_remove.append(n)
G.remove_nodes_from(to_remove)
print("Graph built:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")


#  Random Walk
def recommend_for_user(user_id, top_k, alpha=0.85):
    """
    alpha: probability of continuing walk
    """
    if user_id not in G:
        raise ValueError("User not in graph")

    # Personalized restart vector
    personalization = {node: 0 for node in G.nodes()}
    personalization[user_id] = 1

    paths = nx.generate_random_paths(G, 200, 5, weight="weight", source=user_id)
    recs = set()
    for path in paths:
        # print(path)
        # print(path[5])
        recs.add(path[5])
        # print(len(path))
    # print(len(recs))
    user_books = edges[edges["User-ID"] == user_id]["join_title"].unique()
    recs = recs - set(user_books)
    recs = list(recs)
    # print(recs)
    # print(len(recs))
    if len(recs) > top_k:
        return random.sample(recs, 50)
    return recs

# recommend_for_user("4017", 50)

test_df = pd.read_csv("datasets/test_edges.csv")
test_df["User-ID"] = test_df["User-ID"].astype(str)

output_path = "recommendation_systems/random_walk.csv"
if not os.path.exists(output_path):
    pd.DataFrame(columns=["User-ID", "Recs"]).to_csv(output_path, index=False)


recs_df = pd.DataFrame(columns=["User-ID", "Recs"])
users = [n for n, data in G.nodes(data=True) if data.get("bipartite") == "user"]
recall = 0

for i, user in enumerate(users):
    recs = recommend_for_user(user, 50)
    # print(f"User {user}, Recommendations: {recs}")
    recall += compute_recall(test_df, user, recs)
    print(f"User {user}, cumulative recall: {recall}")

    # Append this row directly to the CSV
    pd.DataFrame([[user, recs]], columns=["User-ID", "Recs"]).to_csv(
        output_path,
        mode="a",
        header=False,
        index=False
    )

print(recall / len(users))