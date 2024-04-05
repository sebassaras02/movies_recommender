import numpy as np
import pandas as pd

def create_df_visualization(docs) -> pd.DataFrame:
    """
    This function creates a dataframe from a list of docs objects.

    Args:
        docs: list of docs objects

    Returns:
        result (pd.DataFrame): dataframe with the metadata and the page content of the docs objects
    """
    # create a dataframe from a docs langchain object 
    result = pd.DataFrame()
    result["Description"] = [doc.page_content for doc in docs]
    # keys of the dictionary to iterate over
    keys = ['authors', 'average_rating', 'num_pages', 'published_year', 'title']
    for key in keys:
        result[key] = [doc.metadata[key] for doc in docs]
    return result[['title', 'authors', 'Description', 'published_year','average_rating','num_pages']]