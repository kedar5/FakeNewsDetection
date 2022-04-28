# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:28:27 2022

@author: Kedar
"""
from keras import backend as K
from tensorflow.keras.models import Model, load_model
import streamlit as st
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd


import plotly.express as px


MODEL_PATH = r"bert_model.pkl"
MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100
# tokenizer_file = "tokenizer.pickle"

wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

model_list = ["Logistic Regression",'Multinomial Naive Bayes Classifier','Gradient Boost Classifier','RFC Classifier']
model_file_list = [r"LR_model.pkl",r"MNVBC_model.pkl",r"GBC_model.pkl",r"RFC_model.pkl"]

# with open(tokenizer_file, 'rb') as handle:
#     tokenizer = pickle.load(handle)

def basic_text_cleaning(line_from_column):
    # This function takes in a string, not a list or an array for the arg line_from_column

    tokenized_doc = word_tokenize(line_from_column)

    new_review = []
    for token in tokenized_doc:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)

    new_term_vector = []
    for word in new_review:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)

    final_doc = []
    for word in new_term_vector:
        final_doc.append(wordnet.lemmatize(word))

    return ' '.join(final_doc)

@st.cache(allow_output_mutation=True)
def Load_model():
    model = load_model(MODEL_PATH)
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session

if __name__ == '__main__':
    st.title('Fake News Finder')
    st.write("We use our expert sources to verify the authenticity of any news you see ")
    st.info("[Sources: Trust me bro]")
    st.subheader("Find out if the news is clickbait")
    sentence = st.text_area("Enter the news title here:", "Some news",height=200)
    predict_btt = st.button("Fake News?")
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://ichef.bbci.co.uk/news/976/cpsprodpb/134A/production/_93583940_037247908-1.jpg");
             background-size: cover;
             background-attachment: scroll;
             background-opacity: 70%;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    model, session = Load_model()
    if predict_btt:

        st.header("Prediction using 4 traditional machine learning model")
        predictions = []
        for model in model_file_list:
            filename = model
            model = pickle.load(open(filename, "rb"))
            prediction = model.predict([sentence])[0]
            predictions.append(prediction)

        dict_prediction = {"Models":model_list,"predictions":predictions}
        df = pd.DataFrame(dict_prediction)

        num_values = df["predictions"].value_counts().tolist()
        num_labels = df["predictions"].value_counts().keys().tolist()

        dict_values = {"true/fake":num_labels,"values":num_values}
        df_prediction = pd.DataFrame(dict_values)
        fig = px.pie(df_prediction, values='values', names='true/fake')
        fig.update_layout(title_text="Comparision between all 4 models: Prediction proportion between True/Fake")
        st.plotly_chart(fig, use_container_width=True)
        st.table(df)
