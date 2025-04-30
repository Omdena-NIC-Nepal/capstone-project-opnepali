#Import BuiltIn Python Module
import streamlit as st #For StreamLit Pagees

def show():
    # NLP demonstration
    st.markdown("## 🔠 NLP Feature Demonstration")

    user_input = st.text_input("**NLP feature demonstration - enter texts related to climate:**")
    if st.button("Display Text Analysis"):
        if user_input == "":
            st.write("Please type some texts.")
        else:
            if user_input:
                from functions_nltk_utility import _analyze_text
                analysis = _analyze_text(user_input)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Tokens") #Tokenization: Input: "NLP is fun!"; Tokens: ["NLP", "is", "fun", "!"]
                    st.write(analysis['tokens'])

                    st.subheader("POS tags") #Part-of-Speech tags: Noun Verb, Adverb etc.
                    st.write(analysis['pos_tags'])
                with col2:
                    st.subheader('Lemmas') #Lemmatization: reduce a word to its lemma: Running to Run
                    st.write(analysis['lemmas'])

                    st.subheader('Entities') #Named Entity: Person, Date/Time, GPE, Location, etc
                    st.write(analysis['entities'])