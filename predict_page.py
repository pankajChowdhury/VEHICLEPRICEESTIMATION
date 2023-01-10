#Importing All Required Dependencies

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import io, misc
import librosa
import tensorflow as tf
# import keras
from scipy.io.wavfile import read
from PIL import Image
import cv2
import pickle
@st.cache
def load_models():
    model = tf.keras.models.load_model('audio_analysis_model.hdf5')
    model_image = tf.keras.models.load_model('image_analysis_model.hdf5')
    with open('saved', 'rb') as file:
      data=pickle.load(file)
    return (model,model_image,data)

models=load_models()


def show_predict_page():
    st.title("Vehicle Price Estimator")

    # MODEL 1 - Takes an Audio Input and Classify whether the defect present is Inner defect, Outer defect or Roller defect

    uploaded_file = st.file_uploader("1. Upload an audio File of the Engine:")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.audio(bytes_data, format='audio/ogg')
        # For reference, example.wav file is  provided here
        frequency = 10000
        def feature_extract(l):
            y = np.array(l)
            sr = frequency
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_scaled = np.mean(mfcc.T, axis=0)

            return mfcc_scaled
        a = read(uploaded_file)

        audio= np.array(a[1],dtype=float)

        test_dummy = feature_extract(audio)

        test_dummy = test_dummy.reshape(1, -1)
        model=models[0]
        ans = model.predict(test_dummy)
        # predicted class is stored in predicted_class
        predicted_class = np.argmax(ans, axis=-1)
        # Stores the parameter implying the defect detected through audio
        audio_defect=0.0
        if predicted_class==0:
          st.subheader('No Defect Detected!')
        elif predicted_class==1:
          st.subheader('Inner Defect Detected!')
          audio_defect=0.2
        elif predicted_class==2:
          st.subheader('Roller Defect Detected!')
          audio_defect=0.2
        else:
          st.subheader('Outer Defect Detected!')
          audio_defect=0.2

    # MODEL 2 - Takes an Image Input and Classify whether the defect present is Minor, Moderate or Severe

    uploaded_file2 = st.file_uploader("2. Upload an Image File of the Vehicle:")
    if uploaded_file2 is not None:
        # Reading the Image
        image = Image.open(uploaded_file2)
        st.image(image, caption='Car Image')

        pic = Image.open(uploaded_file2)
        # pix here denotes the resized image array
        pix = np.asarray(pic)
        pix = cv2.resize(pix, (80, 80))

        im_data=np.array(pix).reshape(80,80,3)
        im_data=im_data/255.0
         # model_image is the image analysis model
        model_image=models[1]
        ans_im=model_image.predict(im_data.reshape(1,80,80,3))
        pred_class_img = np.argmax(ans_im, axis=-1)
        # parameter indicating the defect int the image
        image_defect = 0.0
        if pred_class_img == 0:
          st.subheader('Minor Damage Detected!')
          image_defect=0.2
        elif pred_class_img == 1:
          st.subheader('Moderate Damage Detected!')
          image_defect= 0.3
        elif pred_class_img == 2:
          st.subheader('Severe Damage Detected!')
          image_defect= 0.4

    # The Same Logic can be easilt implemented for a Video Data as a Video is just several Frames of Images together
    uploaded_file3 = st.file_uploader("OR, Upload an Video File of the Vehicle:")
    if uploaded_file3 is not None:

     video_bytes = uploaded_file3.read()
     st.video(video_bytes)


    # MODEL 3 = Decision Tree Regression Model to do Data Analysis as it gave the best result among all

    # Designing all User Input Fields



    # data stores 4 items:
    # 1. A Data Analysis Model (MODEL 3)
    # 2. A Standard Scaler as the input values can be of very different Ranges
    # 3. A HashMap/Dictionary containing Model Name of Vehicle as keys and New Price of Vehicle as values
    # 4. A List of Names of Vehicles
    data=models[2]
    model = data["model"]
    scaling = data["scaling"]
    priceMp = data["priceMap"]
    model_names = data["names"]

    # Taking User Inputs

    col1, col2 = st.columns(2)

    with col1:
        name = st.selectbox( '3. Select Model of the Vehicle:', model_names)

        fuel = st.selectbox(
            '5. Select Fuel Type:',
            ('Petrol', 'Diesel', 'CNG'))

        mileage = st.number_input('7. Enter Mileage in kmpl:')

    with col2:
        location = st.selectbox(
            '4. Select nearest Location:',
            ('Chennai', 'Mumbai', 'Kochi', 'Delhi', 'Coimbatore','Kolkata', 'Jaipur', 'Ahmedabad', 'Hyderabad', 'Pune','Bangalore'))

        owners = st.selectbox(
            '6. Select Number of Owners:',
            ('1','2','3'))
        cc = st.number_input('8. Enter CC of the Engine:')

    years = st.slider('9. Numbers of years used: ', 0, 50, 1)
    kms = st.number_input('10. Kilometers Driven:')

    trans = st.selectbox(
            '11. Select Transmission Type:',
            ('Manual', 'Automatic'))


    # Converting Transmission Type, Location, Fuel into Integer Values
    trans_val=0
    if trans=='Manual' :
        trans_val=0
    else:
        trans_val=1
    new_price=priceMp[name]
    dict_cities = { 'Kochi':0,'Mumbai':1,'Coimbatore':2, 'Hyderabad':3, 'Pune':4, 'Kolkata':5, 'Delhi':6, 'Chennai': 7, 'Jaipur':8, 'Ahmedabad':9, 'Bangalore':10 }
    dict_fuel= {'Petrol':0,'Diesel':1,'CNG':2}
    loc_val = dict_cities[location]
    fuel_val = dict_fuel[fuel]

    # Scaling the Values and Testing the trained Model with it

    inp = np.array([[ loc_val, kms,   fuel_val, trans_val,   mileage,   cc,    new_price, years, owners ]])
    df2 = pd.DataFrame(inp, columns = ['Location','Kilometers_Driven', 'Fuel_Type','Transmission','Mileage', 'Engine', 'New_Price', 'Years', 'Owners'])
    transformed_data= scaling.transform(df2)
    inp=np.array(transformed_data)
    df2 = pd.DataFrame(inp, columns = ['Location','Kilometers_Driven', 'Fuel_Type','Transmission','Mileage', 'Engine', 'New_Price', 'Years', 'Owners'])
    outp = model.predict(df2)


    # Predicting the Final Price using all the outputs of the models

    if st.button('GET ESTIMATED PRICE'):
       ans=float(outp)
       ans = ans - (ans*audio_defect)-(ans*image_defect)
       if ans<=0.0:
          ans=2.2
       st.subheader(f"The Predicted Price is: Rs. {'%.2f'%ans} Lakhs.")
    else:
        st.write('')



