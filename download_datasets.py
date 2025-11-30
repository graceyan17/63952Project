import os
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

dir1 = "datasets/goodreads-book-information-descriptions"
dir2 = "datasets/goodreadsbooks-book-information-pgct"
dir3 = "datasets/book-recommendation-user-reviews"

os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)
os.makedirs(dir3, exist_ok=True)

# goodreads-book-information-descriptions
path = kagglehub.dataset_download("rohitganeshkar/goodreads-book-recommendation-datasets", path=dir1)
print("Path to dataset files:", path)

# goodreadsbooks-book-information-pgct
path = kagglehub.dataset_download("jealousleopard/goodreadsbooks", path=dir2)
print("Path to dataset files:", path)

# book-recommendation-user-reviews
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset", path=dir3)
print("Path to dataset files:", path)