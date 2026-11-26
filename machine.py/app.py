import joblib
import numpy as np
import streamlit as st

st.title("Iris Flower Classifier")
st.write("Sepal aur petal ki values enter karein aur flower ki species jaanein:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    model = joblib.load("iris_svm_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")

    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: **{species[pred]}**")