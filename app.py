import os
import pandas as pd
from pgmpy.readwrite.XMLBIF import XMLBIFReader
from pgmpy.inference import VariableElimination
import streamlit as st


@st.cache_data
def load_dataset():
    df = pd.read_csv(f'data{os.sep}heart.csv')
    df = df[~(df['Cholesterol'] == 0) & ~(df['RestingBP'] == 0)]
    df['ExerciseAngina'] = df['ExerciseAngina'].apply(
        lambda x: False if x == 'N' else True)
    df['HeartDisease'] = df['HeartDisease'].apply(
        lambda x: False if x == 0 else True)

    df['FastingBS'] = df['FastingBS'].apply(
        lambda x: False if x == 0 else True)
    df["Age"] = pd.qcut(x=df["Age"], q=2, labels=["young", "old"])
    df["RestingBP"] = pd.cut(x=df["RestingBP"], bins=[90, 120, 140, 1000], labels=[
        "normal", "high", "very_high"])
    df["Cholesterol"] = pd.cut(x=df["Cholesterol"], bins=[
        0, 200, 240, 1000], labels=["optimal", "borderline", "high"])
    df["MaxHR"] = pd.cut(x=df["MaxHR"], bins=3, labels=[
        "low", "medium", "high"])
    df["Oldpeak"] = pd.cut(x=df["Oldpeak"], bins=3, labels=[
        "low", "medium", "high"])
    return df


@st.cache_data
def load_model():
    return XMLBIFReader(f'model{os.sep}heart_disease_model.xml').get_model()


def exact_inference(model, variables, evidence):
    inference = VariableElimination(model)

    return inference.query(variables=variables, evidence=evidence)


st.title('Heart Disease Risk')

st.write('This is a simple web app to predict the risk of heart disease based on a few parameters.')

df = load_dataset()
model = load_model()
probs = None

LEFT, RIGHT = st.columns(2)

with LEFT:
    left, right = st.columns(2)
    with left:
        age = st.selectbox('Age', df['Age'].unique())
    with right:
        sex = st.selectbox('Sex', df['Sex'].unique())
    with left:
        chest_pain_type = st.selectbox(
            'Chest Pain Type', df['ChestPainType'].unique())
    with right:
        resting_bp = st.selectbox('Resting Blood Pressure',
                                  df['RestingBP'].unique())
    with left:
        cholesterol = st.selectbox('Cholesterol', df['Cholesterol'].unique())
    with right:
        fasting_bs = st.selectbox(
            'Fasting Blood Sugar', df['FastingBS'].unique())
    with left:
        max_hr = st.selectbox('Max Heart Rate', df['MaxHR'].unique())
    with right:
        exercise_angina = st.selectbox(
            'Exercise Angina', df['ExerciseAngina'].unique())
    with left:
        oldpeak = st.selectbox('Oldpeak', df['Oldpeak'].unique())
    with right:
        st_slope = st.selectbox('ST Slope', df['ST_Slope'].unique())

    if st.button('Predict'):
        query = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': str(fasting_bs),
            'MaxHR': max_hr,
            'ExerciseAngina': str(exercise_angina),
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }

        result = exact_inference(
            model, variables=['HeartDisease'], evidence=query)

        probs = result.get_value(
            HeartDisease='True'), result.get_value(HeartDisease='False')

        probs = [round(x, 3) * 100 for x in probs]

with RIGHT:
    if probs:
        st.metric(label='Heart Disease Risk (%)',
                  value=probs[0], delta=probs[1], delta_color='inverse')

        st.write(f'Probability of heart disease: {probs[0]} %')
        st.write(f'Probability of no heart disease: {probs[1]} %')
