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



model_list = ["Logistic Regression",'Multinomial Naive Bayes Classifier','Gradient Boost Classifier','RFC Classifier']
model_file_list = [r"LR_model.pkl",r"MNVBC_model.pkl",r"GBC_model.pkl",r"RFC_model.pkl"]



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
    if predict_btt:

        st.header("Prediction using the 4 models")
        predictions = []
        for model in model_file_list:
            model1 = pickle.load(open(model, "rb"))
            prediction = model1.predict([sentence])[0]
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
