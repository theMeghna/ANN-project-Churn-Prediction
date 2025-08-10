import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from keras.models import load_model
import pickle
import tensorflow as tf
model = tf.keras.models.load_model("model.h5")

with open("le_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)
with open("ohe_geo.pkl", "rb") as f:
    ohe_geo = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

geography=st.selectbox('Geography',ohe_geo.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',18,92)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
number_of_products=st.slider('NumOfProducts',1,4)
has_credit_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'EstimatedSalary':[estimated_salary],
    'NumOfProducts':[number_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],

    
     
})

geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=ohe_geo.get_feature_names_out(['Geography'])
)
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
training_columns = scaler.feature_names_in_
input_data = input_data[training_columns]
input_data_scaled=scaler.transform(input_data)
model_prediction=model.predict(input_data_scaled)
prediction_proba=model_prediction[0][0]
st.write(f"Prediction Probablity: {prediction_proba:.4f}")
if prediction_proba>0.5:
    st.write("The customer is likely to churn.")
else:
    {
    st.write("The customer is not likely to churn.")
}