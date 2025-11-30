import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# goodreads-book-information-descriptions
path = kagglehub.dataset_download("rohitganeshkar/goodreads-book-recommendation-datasets")
print("Path to dataset files:", path)

# goodreadsbooks-book-information-pgct
path = kagglehub.dataset_download("jealousleopard/goodreadsbooks")
print("Path to dataset files:", path)

# book-recommendation-user-reviews
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print("Path to dataset files:", path)