#Importing All Required Dependencies
import streamlit as st
from explore_page import show_explore_page
from predict_page import show_predict_page



page = st.sidebar.selectbox("Pages", ("Vehicle Price Estimator", "Explore More"))
if page=="Vehicle Price Estimator":
         show_predict_page()

else:
         show_explore_page()


