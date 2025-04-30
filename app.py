# Import custom functions from local python (.py) files
from functions_data_utility import _data_load 

#Import Standard Libraries
import streamlit as st #To build and share web apps for data science and machine learning projects
import nltk #Natural Language Toolkit for Tokenization, Stopword removal, Part of Speech (POS) Tagging etc.

try:
    #Download the Punkt tokenizer models, which are pre-trained data used by NLTK’s tokenizers to split text into sentences and words.
    #Use case: Word/Sentence Tokenization
    nltk.download('punkt') # Needed for tokenization
    
    #Download the Brown Corpus, the first million-word electronic corpus of English.
    #Use case: Useful for linguistic analysis, text classification, and POS tagging.
    #Contains: 500+ samples of English text from 15 categories (news, editorial, romance, etc.).
    nltk.download('brown') 

    #Download WordNet, a large lexical database of English.
    #Use case: Used for lemmatization, synonym/antonym detection, and word sense disambiguation.
    nltk.download('wordnet')

    #Download a pre-trained part-of-speech (POS) tagger.
    #Use case: Tags each word in a sentence with its POS (noun, verb, adjective, etc.).
    nltk.download('averaged_perceptron_tagger')

    #Download the Movie Reviews Corpus, commonly used for sentiment analysis.
    #Use case: Text classification, training models to understand positive vs. negative reviews.
    #Structure: 1000 positive and 1000 negative movie reviews.
    nltk.download('movie_reviews')

    #Download the CoNLL 2000 corpus, used for chunking tasks (also called shallow parsing).
    #Use case: Helps with tasks like identifying noun phrases (NP) and verb phrases (VP).
    nltk.download('conll2000')

    print("Download successful: NLTK corpora.")
except Exception as e:
    print(f"Download unsuccessful: NLTK corpora: {e}")

#Define the Main function from where the App invokes
def main():
    st.set_page_config(
        page_title = "Capstone-Climate Change Impact Assessment and Prediction System for Nepal",
        page_icon = "⛅",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    # Load data once and cache it
    _df_data = _data_load()

    # Store data in session state for access across pages
    if 'data' not in st.session_state:
        st.session_state.data = _df_data

    # Define Navigation Items in Sidebar Navigation
    with st.sidebar:
        st.title('Navigate to')
        page = st.radio(
            "===================================",
            options=[
                "Home",
                "Data Preprocess",
                "Exploratory Data Analysis (EDA)", 
                "Feature Engineering",
                "Model Training & Evaluation",
                "Temperature Prediction",
                "Basic Text Analysis",
                "Climate Text Analysis"
            ],
            # Add key to prevent duplicate widgets
            key="sidebar_nav_radio"  
        ) 

    # Display the selected page
    if page == "Home":
        from utils import _Home
        #from pages import Home
        st.session_state.current_page = "Home"
        _Home.show()
    elif page == "Data Preprocess":
        from utils import _Data_Preprocess
        #from pages import Home
        st.session_state.current_page = "Data Preprocess"
        _Data_Preprocess.show()
    elif page == "Exploratory Data Analysis (EDA)":
        from utils import _EDA
        st.session_state.current_page = "Exploratory Data Analysis (EDA)"
        _EDA.show()
    elif page == "Feature Engineering":
        from utils import _Feature_Engineering
        st.session_state.current_page = "Feature Engineering"
        _Feature_Engineering.show()
    elif page == "Model Training & Evaluation":
        from utils import _Model_TrainingEvaluation
        st.session_state.current_page = "Model Training & Evaluation"
        _Model_TrainingEvaluation.show()
    elif page == "Temperature Prediction":
        from utils import _Prediction
        st.session_state.current_page = "Temperature Prediction"
        _Prediction.show()
    elif page == "Basic Text Analysis":
        from utils import _Basic_Text_Analysis
        st.session_state.current_page = "Basic Text Analysis"
        _Basic_Text_Analysis.show()
    elif page == "Climate Text Analysis":
        from utils import _Climate_Text_Analysis
        st.session_state.current_page = "Climate Text Analysis"
        _Climate_Text_Analysis.show()
    
#Prevent calling of main() from if imported elsewhere.
if __name__ == "__main__":
    main()  

#__name__ above is a special built-in variable in Python. When we run the script directly (e.g., python 
#app.py), then __name__ is set to __main__. However when we import the script into another file 
#(like a module), __name__ will be set to the name of the file (e.g., my_app), not __main__.

