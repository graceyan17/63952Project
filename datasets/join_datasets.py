import pandas as pd
import numpy as np

def normalize(x):
    return (
        x.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
    )

book_reviews = pd.read_csv("datasets/book-recommendation-user-reviews/Books.csv")
book_descriptions = pd.read_csv("datasets/goodreads-book-information-descriptions/book_data.csv")
book_pg_ct = pd.read_csv("datasets/goodreadsbooks-book-information-pgct/books.csv",
                         on_bad_lines="skip")

book_reviews["join_title"]  = normalize(book_reviews["Book-Title"])
book_reviews_titles = book_reviews["join_title"].unique()
book_reviews["join_author"] = normalize(book_reviews["Book-Author"])

book_descriptions["join_title"] = normalize(book_descriptions["book_title"])
book_descriptions_titles = book_descriptions["join_title"].unique()
book_descriptions["join_author"] = normalize(book_descriptions["book_authors"])

book_pg_ct["join_title"] = normalize(book_pg_ct["title"])
book_pg_ct_titles = book_pg_ct["join_title"].unique()
book_pg_ct["join_author"] = normalize(book_pg_ct["authors"])

full_info_book_list = np.intersect1d(book_reviews_titles, 
                                     np.intersect1d(book_descriptions_titles, 
                                                    book_pg_ct_titles))

isbns = book_reviews.loc[
    book_reviews["join_title"].isin(full_info_book_list),
    "ISBN"].astype(str).tolist()
print(len(isbns))

ratings = pd.read_csv("datasets/book-recommendation-user-reviews/Ratings.csv")
ratings["ISBN"] = ratings["ISBN"].astype(str)
print(ratings["ISBN"].isin(isbns).sum())

# Convert all ISBNs to strings
book_reviews["ISBN"] = book_reviews["ISBN"].astype(str)
book_descriptions["book_isbn"] = book_descriptions["book_isbn"].astype(str)
book_pg_ct["isbn"] = book_pg_ct["isbn"].astype(str)
book_pg_ct["isbn13"] = book_pg_ct["isbn13"].astype(str)

complete_csv = ratings[ratings["ISBN"].isin(isbns)].copy()

users = pd.read_csv("datasets/book-recommendation-user-reviews/Users.csv")
complete_csv = complete_csv.merge(users, on="User-ID", how="left")
complete_csv = complete_csv.merge(
    book_reviews[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication",
       "Publisher", "join_title"]],
    on="ISBN",
    how="left"
)
complete_csv = complete_csv.merge(
    book_descriptions[["join_title", "book_desc", "book_edition", "book_format",
       "book_ pages", "book_rating", "book_rating_count",
       "book_review_count", "genres", "book_ price"]],
    on="join_title",
    how="left"
)
complete_csv = complete_csv.merge(
    book_pg_ct[["join_title", "language_code"]],
    on="join_title",
    how="left"
)

complete_csv = complete_csv.rename(columns={
    "Location": "User_Location",
    "Age": "User_Age",
    "book_ pages": "book_pages",
    "book_ price": "book_price",
})

complete_csv = complete_csv.drop_duplicates(subset=["User-ID", "ISBN"])
print(len(complete_csv))
complete_csv.to_csv("datasets/joined_complete_info.csv", index=False)