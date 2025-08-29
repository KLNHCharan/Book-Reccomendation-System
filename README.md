## Book Recommendation System

This project is a content-based book recommendation system built as a web application using Streamlit. It allows users to select a book from a given dataset and receive a list of similar titles based on shared attributes.

### Features

  * **Dynamic Recommendations:** Provides real-time book suggestions from a dataset of over 2,000 books.
  * **Content-Based Filtering:** The recommendation engine analyzes book titles and authors to identify similarities.
  * **Intuitive UI:** Features a clean, interactive user interface built with Streamlit, making it easy to select a book and view recommendations.

### Technology Stack

  * **Python:** The core programming language.
  * **Streamlit:** Used to create the web application's front-end.
  * **Pandas:** For data loading, preprocessing, and manipulation.
  * **Scikit-learn:** Provides the machine learning tools for the recommendation engine, specifically `TfidfVectorizer` and `cosine_similarity`.

### Methodology

The recommendation system uses a two-step process to generate recommendations:

1.  **Feature Extraction:** The `TfidfVectorizer` from scikit-learn converts the text data (book titles and authors) into a numerical matrix. This process identifies the importance of words in the dataset.
2.  **Similarity Calculation:** The `cosine_similarity` algorithm then calculates a similarity score for every book pair. The system then recommends the books with the highest scores relative to the user's selected title.

### Getting Started

#### Prerequisites

  * Python 3.8 or higher
  * pip (Python package installer)

#### Installation

1.  Clone this repository to your local machine.
2.  Navigate to the project directory.
3.  Install the required libraries:
    ```bash
    pip install streamlit pandas scikit-learn
    ```

#### Running the Application

1.  Ensure the `books.csv` file is in the same directory as `app.py`.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.
