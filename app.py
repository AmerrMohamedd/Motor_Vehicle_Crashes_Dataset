import streamlit as st
import pandas as pd
from joblib import load

lr_model = load('linear_model.pkl')
poly_model = load('poly_model.pkl')
knn_model = load('knn_model.pkl')
dt_model = load('tree_model.pkl')
poly_features = load('poly_features.pkl')

def make_prediction(model, input_data, poly=False):
    if poly:
        input_data = poly_features.transform(input_data)
    prediction = model.predict(input_data)
    return prediction

st.title('Regression Model Deployment')

st.sidebar.header('Input Parameters')

latitude = st.sidebar.number_input('Latitude', format="%.6f")
longitude = st.sidebar.number_input('Longitude', format="%.6f")
persons_killed = st.sidebar.number_input('Number of persons killed', value=0, min_value=0)
pedestrians_injured = st.sidebar.number_input('Number of pedestrians injured', value=0, min_value=0)
pedestrians_killed = st.sidebar.number_input('Number of pedestrians killed', value=0, min_value=0)
cyclists_injured = st.sidebar.number_input('Number of cyclists injured', value=0, min_value=0)
cyclists_killed = st.sidebar.number_input('Number of cyclists killed', value=0, min_value=0)
motorists_injured = st.sidebar.number_input('Number of motorists injured', value=0, min_value=0)
motorists_killed = st.sidebar.number_input('Number of motorists killed', value=0, min_value=0)

input_data = pd.DataFrame({
    'latitude': [latitude],
    'longitude': [longitude],
    'number of persons killed': [persons_killed],
    'number of pedestrians injured': [pedestrians_injured],
    'number of pedestrians killed': [pedestrians_killed],
    'number of cyclist injured': [cyclists_injured],
    'number of cyclist killed': [cyclists_killed],
    'number of motorist injured': [motorists_injured],
    'number of motorist killed': [motorists_killed]
})

lr_prediction = make_prediction(lr_model, input_data)
poly_prediction = make_prediction(poly_model, input_data, poly=True)
knn_prediction = make_prediction(knn_model, input_data)
dt_prediction = make_prediction(dt_model, input_data)

st.subheader('Predictions')
st.write('Linear Regression:', lr_prediction[0])
st.write('Polynomial Regression:', poly_prediction[0])
st.write('KNN Regression:', knn_prediction[0])
st.write('Decision Tree Regression:', dt_prediction[0])