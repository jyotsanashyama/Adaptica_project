import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))
review = st.text_input('Tell Us What You Think! Your feedback helps us improve.')

if st.button('Let me think!'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write("Thanks for your honest feedback! We are always striving to improve. While your experience wasn't perfect, your insights help us make things better. Would you like to chat with our friendly AI assistant and share some suggestions?")

       
    else:
        st.write("We're thrilled to hear you had a positive experience! Your feedback motivates us to keep building something great. How about taking your interaction a step further and exploring our AI assistant to see what innovative things we can do together?")
       
    
if st.button('AI Pal'):
   
   
        chatbot_link = "https://your_chatbot_url"  # Replace with your actual chatbot URL
        st.button("Hey! My Friendo", on_click=lambda: st.sidebar.open(chatbot_link))
    
       
