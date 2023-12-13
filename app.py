import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json

st.header("Prediksi WanPrestasi")
st.image('default.jpg')
st.subheader("""
Nama: Adam Nur Ramadan

Batch: HCK-010

Objective: Membuat model klasifikasi untuk memprediksi pembayaran bulan selanjutnya
""")
# Data Loading
model = joblib.load('model.pkl')
df = pd.read_csv('dataset.csv')
feature = json.load(open('feature.json','r'))

# Header
st.header("Data Frame")
st.write(df)

#func
def user_input():
    inputter = {}
    for i in feature:
        if 'education' not in i :
            if 'amt' in i:
                inputter[i] = st.sidebar.slider(i,-1000,1000,1)
            else:
                inputter[i] = st.sidebar.slider(i,-10,10,1)
    inputter[feature[len(feature)-1]] = st.sidebar.slider(feature[len(feature)-1],0,4,2)
    inputter = pd.DataFrame([inputter])
    return inputter

# Sidebar Input
input = user_input()

st.subheader('Input')
st.write(input)

if st.button('predict'):
    prediction = model.predict(input)
    
    if prediction == 1:
        prediction = 'Bukan Wanprestasi'
    else:
        prediction = 'Wanprestasi'
    
    st.header('Dari data tersebut, User adalah : ' + prediction)
    
st.header("Kesimpulan")
st.write("Data 'default_payment_next_month' dipengaruhi oleh data pay, pay_amt dan education_level, dan model SVC dengan kernel linear adalah model yang cocok untuk memprediksi 'default_payment_next_month'.")
    
