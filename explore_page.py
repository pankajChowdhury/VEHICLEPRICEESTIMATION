# Importing All Dependencies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

@st.cache
def load_data():
    train_df = pd.read_csv('train-data.csv')
    test_df = pd.read_csv('test-data.csv')
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    frames = [train_df, test_df]
    df = pd.concat(frames)
    df = df.dropna()
    df['New_Price'] = df['New_Price'].str.extract('(\d+)').astype(float)
    df['Decrease'] = df['New_Price']-df['Price']
    df = df.drop(['Unnamed: 0', 'Name', 'Power', 'Seats'], axis=1)
    df['Years'] = 2022 - df['Year']
    return df

df=load_data()
def show_explore_page():
    st.title("Explore Datasets Used for Model Development")
    st.subheader("1.Audio Analysis Model:")
    st.write(
        'The Dataset used for this model contains audio input of Acoustic Emission of Engine.Then mfcc was used for feature extraction and used this data to train the model.')
    st.write('Example of Audio:')
    audio_file = open('car.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    st.subheader("2.Image Analysis Model:")
    st.write(
        'The Dataset used for this model contains images of Vehicles in different Condition-Severe Damage, Moderate Damage and Minor Damage. ')
    st.write('The Images were converted into Numpy Arrays for training of the Model')
    st.write('Example of Images:')
    col1, col2, col3 = st.columns(3)

    with col1:
        image = Image.open('severe.JPEG')
        st.image(image, caption='Severe Damage')


    with col2:
        image = Image.open('moderate.JPEG')
        st.image(image, caption='Moderate Damage')

    with col3:
        image = Image.open('minor.JPEG')
        st.image(image, caption='Minor Damage')



    st.subheader("3.Data Analysis Model:")
    st.write(
        "Several models was used over the dataset but the solution that gave best result to this Regression Problem was Decision Tree Regression.")
    st.write("Some Details about the DataSet is given Below")


    data = df["Location"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    st.write("""#### Percentage of Data from Different Cities""")
    st.pyplot(fig1)

    st.write("""#### Avg Price of Old Vehicles(in Lakhs) in Different Cities""")
    data = df.groupby(["Location"])["Price"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Avg Decrease in Price(in Lakhs) Based on Years of Use""")
    data = df.groupby(["Years"])["Decrease"].mean().sort_values(ascending=True)
    st.line_chart(data)




