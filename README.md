# iris_svm_streamlit
# Iris Flower Classifier (Streamlit + SVM)

A local web application built with **Streamlit** and **Scikit-Learn** that predicts the species of an Iris flower (**Setosa**, **Versicolor**, or **Virginica**) from four input features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Project Files
- `machine.py` — Streamlit application interface and prediction logic
- `iris_svm_model.pkl` — Trained SVM classification model
- `iris_scaler.pkl` — Fitted feature scaler
- `requirements.txt` — Required Python packages

## How to Run Locally
1. Open a terminal inside the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   'streamlit run app.py 
