import streamlit as st
import pandas as pd
import numpy  as np
import joblib
from tensorflow .keras.models import load_model

st.header('StyleEase Customer Database')
st.write("""
The provided StyleEase customer database is used for predicting which customers are more likely to churn.
""")

@st.cache_data
def fetch_data():
    df = pd.read_csv('churn.csv')
    return df

df = fetch_data()

feedback = st.selectbox('feedback', df['feedback'].unique())
points_in_wallet = st.number_input('points_in_wallet', value=0)
avg_transaction_value = st.number_input('avg_transaction_value', value=0)
membership_category = st.selectbox('membership_category', df['membership_category'].unique())

data = {
    'feedback' : feedback,
    'points_in_wallet' : points_in_wallet,
    'avg_transaction_value' : avg_transaction_value,
    'membership_category' : membership_category
}
input = pd.DataFrame(data, index=[0])

st.subheader('Predict')
st.write(input)

load_model = load_model("ann_tuned_model.h5")
transform = joblib.load('prepro_styleease.pkl')  

if st.button('Predict'):
    change = transform.transform(input)
    pred = load_model.predict(change)
    results = np.where(pred >= 0.5, 1, 0)

    if results == 1:
        results = 'Churn'
    else:
        results = 'Does not churn'

    st.write('Based on the input, the placement model predicted: ')
    st.write(results)