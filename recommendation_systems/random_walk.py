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
df["ISBN"] = df["ISBN"].astype(str)
edges["User-ID"] = edges["User-ID"].astype(str)
edges["ISBN"] = edges["ISBN"].astype(str)

# ==========================
#  Build bipartite graph
# ==========================
G = nx.Graph()

# Add user nodes and book nodes with type label
users = df["User-ID"].unique()
books = df["ISBN"].unique()

G.add_nodes_from(users, bipartite="user")
G.add_nodes_from(books, bipartite="item")

# Add edges user <-> book with weight = rating
for _, row in edges.iterrows():
    G.add_edge(row["User-ID"], row["ISBN"], weight=int(row["Book-Rating"]))

print("Graph built:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")


#  Personalized PageRank (Random Walk with Restart)
def recommend_for_user(user_id, top_k=10, alpha=0.85):
    """
    Restart at user_id with prob alpha
    Random walk through graph otherwise
    """
    if user_id not in G:
        raise ValueError("User not in graph")

    # Personalized restart vector
    personalization = {node: 0 for node in G.nodes()}
    personalization[user_id] = 1

    # Compute personalized PageRank
    pr = nx.pagerank(
        G,
        alpha=1 - alpha,
        personalization=personalization,
        weight="weight"
    )

    # Convert dict → sorted list
    pr_series = pd.Series(pr)

    # Only keep book nodes
    user_books = df[df["User-ID"] == user_id]["ISBN"].unique()
    pr_books = pr_series.drop(users)  # drop user nodes

    # Remove books the user already rated
    pr_books = pr_books.drop(user_books, errors="ignore")

    # Top recommendations
    return pr_books.sort_values(ascending=False).head(top_k).index.tolist()


test_df = pd.read_csv("datasets/test_edges.csv")
test_df["User-ID"] = test_df["User-ID"].astype(str)
test_df["ISBN"] = test_df["ISBN"].astype(str)

recall = 0
ct = 0
for user in users:
    recs = recommend_for_user(user, top_k=20)
    recall += compute_recall(test_df, user, recs)
    ct += 1

print(recall / ct)