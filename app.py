import streamlit as st
import joblib
import pandas as pd
import sklearn


try:
    model = joblib.load('incre_model.h5')
    scaler = joblib.load('incre_scaler.h5')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

def predict(features):
    try:
       
        X = pd.DataFrame([features])
        
       
        X_scaled = scaler.transform(X)
        
 
        prediction = model.predict(X_scaled)[0]
    except Exception as e:
        return f"Error making prediction: {e}"

    return prediction

def main():
    st.title('Machine Learning Model Deployment')
    st.write('This app demonstrates the deployment of a machine learning model.')

    st.subheader('Input Features')
    borough = st.number_input('Borough', value=2)
    latitude = st.number_input('Latitude', value=40.7128)
    longitude = st.number_input('Longitude', value=-74.0060)
    on_street_name = st.text_input('On Street Name', value='1')
    cross_street_name = st.text_input('Cross Street Name', value='2')
    persons_killed = st.number_input('Number of Persons Killed', value=0)
    pedestrians_injured = st.number_input('Number of Pedestrians Injured', value=1)
    pedestrians_killed = st.number_input('Number of Pedestrians Killed', value=0)
    cyclists_injured = st.number_input('Number of Cyclists Injured', value=0)
    cyclists_killed = st.number_input('Number of Cyclists Killed', value=0)
    motorists_injured = st.number_input('Number of Motorists Injured', value=0)
    motorists_killed = st.number_input('Number of Motorists Killed', value=0)
    contributing_factor_vehicle_1 = st.text_input('Contributing Factor Vehicle 1', value='3')
    contributing_factor_vehicle_2 = st.text_input('Contributing Factor Vehicle 2', value='1')
    vehicle_type_code_1 = st.text_input('Vehicle Type Code 1', value='4')
    vehicle_type_code_2 = st.text_input('Vehicle Type Code 2', value='2')

    features = {
        'borough': borough,
        'latitude': latitude,
        'longitude': longitude,
        'on street name': on_street_name,
        'cross street name': cross_street_name,
        'number of persons killed': persons_killed,
        'number of pedestrians injured': pedestrians_injured,
        'number of pedestrians killed': pedestrians_killed,
        'number of cyclist injured': cyclists_injured,
        'number of cyclist killed': cyclists_killed,
        'number of motorist injured': motorists_injured,
        'number of motorist killed': motorists_killed,
        'contributing factor vehicle 1': contributing_factor_vehicle_1,
        'contributing factor vehicle 2': contributing_factor_vehicle_2,
        'vehicle type code 1': vehicle_type_code_1,
        'vehicle type code 2': vehicle_type_code_2
    }

    if st.button('Predict'):
        prediction = predict(features)
        st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
