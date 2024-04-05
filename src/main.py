import streamlit as st
import numpy as np
import pandas as pd
from utils.similarity_search import create_retriever, search_similar_books
from utils.visualization import create_df_visualization


st.title('Movie Recommender System')
st.markdown('This app is a simple movie recommender system based on the text you provide. The system will search for similar movies based on the text you provide. You can also filter the search by category.')

with st.sidebar:
    st.subheader('About the App')
    st.markdown('Data used for the training come from the following source: https://www.kaggle.com/datasets/tanguypledel/science-fiction-books-subgenres')
    st.markdown('The recommendation system is based on Ada002 model from OpenAI. The embeddings are stored in Pinecone. The search is done using the cosine similarity.')
    st.empty()
    st.subheader('Author')
    st.markdown('Sebastián Sarasti Zambonino')
    st.markdown('Data Scientist - Machine Learning Engineer')
    st.markdown('https://www.linkedin.com/in/sebastiansarasti/')
    st.markdown('https://github.com/sebassaras02')

df = pd.read_csv('books.csv')

categories = list(set(df["categories"]))

col1, col2 = st.columns(2)

with col1:
    text = st.text_area("Enter a text")

with col2:
    option_cat = st.selectbox("Pick a category", categories)

button = st.button("Search")

if button and option_cat and text:
    retriever = create_retriever()
    docs = search_similar_books(retriever, text, metadata=option_cat, use_metadata=True)
    df_final = create_df_visualization(docs)
    st.subheader("Books similar to the text given")
    st.dataframe(df_final)
elif button and text:
    retriever = create_retriever()
    docs = search_similar_books(retriever, text)
    df_final = create_df_visualization(docs)
    st.subheader("Books similar to the text given")
    st.dataframe(df_final)




