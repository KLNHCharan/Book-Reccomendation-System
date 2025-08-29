import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Streamlit Page Configuration ---
st.set_page_config(layout="centered")

# --- Load and Prepare Data ---
try:
    # Fill any missing author or title data to prevent errors
    books_df = pd.read_csv("C:/Users/klnhc/Downloads/books-recommendation-system-ui/books-recommendation-system-ui/books.csv", on_bad_lines='skip')
    books_df['authors'] = books_df['authors'].fillna('')
    books_df['title'] = books_df['title'].fillna('')

    # Create a 'tags' column for content-based recommendation
    # We combine title and authors as a basis for similarity
    books_df['tags'] = books_df['title'] + ' ' + books_df['authors']
except FileNotFoundError:
    st.error("The 'books.csv' file was not found. Please make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# --- Recommendation Logic ---
# Use TF-IDF to convert the book tags into a matrix of token counts.
# This helps in finding similar books based on shared titles and authors.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['tags'])

# Calculate the cosine similarity between each book and all other books.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series of book titles to easily get an index from a title.
# We reset the index to ensure it aligns with the cosine similarity matrix
books_df = books_df.reset_index(drop=True)
book_titles = pd.Series(books_df.index, index=books_df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=books_df):
    """
    Finds and returns the top 5 most similar books based on title and authors.
    """
    if title not in book_titles.index:
        return pd.DataFrame()

    # Get the index of the book that matches the title
    idx = book_titles[title]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 6 most similar books (the first one is the book itself)
    sim_scores = sim_scores[1:6]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar books
    return df.iloc[book_indices][['title', 'authors']]

# --- Streamlit UI ---
st.title('Simple Book Recommendation System')
st.markdown("""
<style>
.st-emotion-cache-1px0v1j.e1f1d6gn2 {
    text-align: center;
}
.st-emotion-cache-10w473j.e1f1d6gn0 {
    font-size: 20px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.header('Find your next favorite read!')

# Dropdown menu for book selection
book_list = books_df['title'].tolist()
selected_book = st.selectbox(
    'Select a book:',
    book_list
)

# Button to trigger recommendation
if st.button('Get Recommendations'):
    st.subheader(f'Recommendations for "{selected_book}":')
    recommendations = get_recommendations(selected_book)

    if not recommendations.empty:
        # Display the recommendations
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
    else:
        st.warning(f'Could not find recommendations for "{selected_book}". Please try another book.')

st.markdown("---")
st.write("This is a simple content-based recommendation system that uses book titles and authors from the provided CSV file to suggest similar books.")
