# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22  17:00:22 2021

@co-author: Reckon Mazumdar, Akash Chetia, Kunjal Sarma, Srimanjyoti Dutta, Samarjit Sharma
"""
##Importing required libraries
import streamlit as st
import pickle
import nltk
import string
import requests, uuid
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from streamlit.proto.Markdown_pb2 import Markdown
ps = PorterStemmer()
#API keys
from dotenv import load_dotenv
import os
load_dotenv() 
API_KEY=os.getenv("API_KEY")
LOCATION=os.getenv("LOCATION")


#Page name and icon
PAGE_CONFIG = {"page_title":"Assamese sentiment analyzer", "page_icon":"Asset/asset01.jpeg","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


#Re-set Hamburger menu and footer note-"Made with ****" removing hack
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


#footer styles  put on top to load first to incorporate rest load
footer="""<style>
a:link , a:visited{
color: grey;
background-color: white;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: white;
text-decoration: underline;
}
.css-hi6a2p{
    padding:0 10px;
}
.footer {
width: 100%;
background-color:  white;
color: grey;
text-align: center;
margin-top:100px;
}
</style>
<div class="footer">
<p>Developed with ❤ by - <br>
<a style='text-align: center;' href="https://www.linkedin.com/in/reckon-mazumdar/" target="_blank">Reckon Mazumdar</a>| 
<a style='text-align: center;' href="https://www.linkedin.com/in/akash-chetia" target="_blank">Akash Chetia</a>| 
<a style='text-align: center;' href="https://www.linkedin.com/in/kunjalsarma" target="_blank">Kunjal Sarma</a>|
<a style='text-align: center;' href="https://www.linkedin.com/in/srimanjyoti-dutta-74317b154" target="_blank">Srimanjyoti Dutta</a>|
<a style='text-align: center;' href="https://www.linkedin.com/in/samarjit-sharma-7b55371aa" target="_blank">Samarjit Sharma</a>
</p>
</div>

"""



#Header image grid pattern
col1, col2, col3 = st.beta_columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image("Asset/asset02.png")

with col3:
    st.write("")

#App title
st.markdown("<h1 style='text-align: center; color: red;'>Assamese song review sentiment analyzer</h1>", unsafe_allow_html=True)

examples= """
### Few Assamese sentences to test our app
##### Negative sentences
- as : এই যুগৰ আটাতকৈ বেয়া গায়ক
> en : The worst singers of this era
- as : মোৰ কাণৰ বাবে নিখুঁত অত্যাচাৰ।
> en : Perfect torture for my ears.
##### Positive sentences
- as: এই গানটো ইমান আৰামদায়ক
> en : This song is so comfortable
- as : বহুত ভাল লাগে
> en : It's very nice
"""
st.markdown(examples, unsafe_allow_html=True)
# --------------------In language sentimental analysis ------------------------------

#Function to clean the text
def transform_text_inl(text):
    text=nltk.word_tokenize(text)
    stop=['অতএব', 'অথচ', 'অথবা', 'অধঃ', 'অন্ততঃ', 'অৰ্থাৎ', 'অৰ্থে', 'আও', 'আঃ', 'আচ্ছা', 'আপাততঃ', 'আয়ৈ', 'আৰু',
      'আস্', 'আহা', 'আহাহা', 'ইতস্ততঃ', 'ইতি', 'ইত্যাদি', 'ইস্', 'ইহ', 'উঃ', 'উৱা', 'উস্', 'এতেকে', 'এথোন',
      'ঐ', 'ওঁ', 'ওৰফে', 'ঔচ্', 'কি', 'কিম্বা', 'কিন্তু', 'কিয়নো', 'কেলেই', 'চোন', 'ছাৰি', 'ছিকৌ', 'ছেই',
      'ঠাহ্', 'ঢেঁট্', 'তত', 'ততক', 'ততেক', 'তেতেক', 'ততেক', 'তত্ৰাচ', 'তথা', 'তথৈবচ', 'তাতে', 'তেও',
      'তো', 'তৌৱা', 'দেই', 'দেহি', 'দ্বাৰা', 'ধৰি', 'ধিক্', 'নতুবা', 'নি', 'নো', 'নৌ', 'পৰা', 'পৰ্যন্ত',
      'বৰঞ্চ', 'বহিঃ', 'বাবে', 'বাৰু', 'বাহ্', 'বাহিৰে', 'বিনে', 'বে', 'মতে', 'যথা', 'যদি', 'যদ্যপি', 'যে',
      'যেনিবা', 'যেনে', 'যোগে', 'লৈ', 'সত্ত্বে', 'সমন্ধি', 'সম্প্ৰতি', 'সহ', 'সু', 'সেইদেখি', 'সৈতে', 'স্বতঃ', 'হঞে', 'হতুৱা', 'হন্তে',
      'হবলা', 'হয়', 'হা', 'হুঁ', 'হুই', 'হে', 'হেই', 'হেঃ', 'হেতুকে', 'হেনে', 'হেনো', 'হেৰ', 'হেৰি', 'হৈ', 'হোঁ', 'ইঃ', 'ইচ্',
      'চুহ্', 'চুঃ', 'আঁ']
    punc="!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~।"
    puncword=[]
    for i in punc:
         puncword.append(i)
    y=[]
    for i in text:
        if i not in stop and i not in punc:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        i=''.join(j for j in i if not j in puncword)
        y.append(i)
    return " ".join(y)

#Loading the model and vectorizer
tfidf = pickle.load(open('vectorizer_inl.pkl','rb'))
model_inl = pickle.load(open('model_inl.pkl','rb'))

st.subheader('In language prediction:')
st.markdown('`This approach is based on training the classifiers on the same language as text.`')
st.markdown('**Gives 81% accurate results**.')
#Text box
ip_sentence=st.text_area("Enter the assamese sentence..")

if st.button('Predict.'):
    #Taking input and Cleaning the text
    transformed_sentence=transform_text_inl(ip_sentence)
    #Vectorizing
    vec=tfidf.transform([transformed_sentence])
    #predicting result and displaying it
    result= model_inl.predict(vec)[0]
    if result == 1:
        st.header("Positive😇")
    else:
        st.header("Negative☹️")

# ------------------------------------------------------------------------

# --------------------Machine translation sentimental analysis -----------
#Function to clean the text
def transform_text_mt(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

#Function to translate the text
def translate(text):
    # Add your subscription key and endpoint
    subscription_key = API_KEY
    endpoint = "https://api.cognitive.microsofttranslator.com"

    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = LOCATION

    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'as',
        'to': ['en']
    }
    constructed_url = endpoint + path

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': text
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

#Loading the model and vectorizer
cv = pickle.load(open('vectorizer_mt.pkl','rb'))
model_mt = pickle.load(open('model_mt.pkl','rb'))

st.subheader("Machine translation based prediction:")
st.markdown('`In this approach we train the classifier on English reviews and for testing, we translate the Assamese reviews into English using Microsoft Translator api and then we classify the Sentiment of the review.`') 
st.markdown('**Gives 88% accurate results**.')
#Text box
ip_sentence2=st.text_area("Enter the assamese sentence")
translated_text=translate(ip_sentence2)

if st.button('Predict'):
    #Taking input and Cleaning the text
    transformed_sentence2=transform_text_mt(translated_text)
    #Vectorizing
    vec2=cv.transform([transformed_sentence2])
    #predicting result and displaying it
    result2= model_mt.predict(vec2)[0]
    st.text("Translated text : {data}".format(data=translated_text))
    if result2 == 1:
        st.header("Positive😇")
    else:
        st.header("Negative☹️")
        
# ------------------------------------------------------------------------

st.markdown(footer,unsafe_allow_html=True)