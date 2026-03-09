import joblib
import pandas as pd
import streamlit as st

import os

# Load the model pipeline using a robust path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'model_pipeline.pkl')
model_pipeline = joblib.load(model_path)

st.title("Titanic Survival Predictor")
st.write("Enter your details to find out if you could have survived the Titanic.")

# User inputs
pclass_mapping = {"1st Class (Upper)": 1, "2nd Class (Middle)": 2, "3rd Class (Lower)": 3}
pclass_input = st.selectbox("Passenger Class", list(pclass_mapping.keys()))
pclass = pclass_mapping[pclass_input]
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=20, value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=20, value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.2, step=1.0)

# Simpler term for Boarding Port
embarked_mapping = {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"}
embarked_input = st.selectbox("Where did you board the Titanic?", list(embarked_mapping.keys()))
embarked = embarked_mapping[embarked_input]

if st.button("Predict Survival"):
    # Calculate family size
    fsize = sibsp + parch
    
    # Create input DataFrame with columns expected by X
    # X has columns: Pclass, Sex, Age, Fare, Embarked, fsize
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'Fare': fare,
        'Embarked': embarked,
        'fsize': fsize
    }])
    
    # Make prediction
    try:
        prediction = model_pipeline.predict(input_data)[0]
        
        if prediction == 1:
            st.success("🎉 You would have likely SURVIVED!")
        else:
            st.error("💀 You would have likely NOT SURVIVED.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
