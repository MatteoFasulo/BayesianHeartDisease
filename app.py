import os
import numpy as np
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


st.set_page_config(
    page_title="Heart Disease Risk",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/MatteoFasulo/BayesianClassifier',
        'Report a bug': 'https://github.com/MatteoFasulo/BayesianClassifier/issues',
        'About': 'This is a simple Web App to predict the risk of heart disease based on a few parameters of the patient. The model is a Bayesian Network and the inference is performed using Variable Elimination.'
    }
)

st.title('Heart Disease Risk')

st.subheader('Heart Disease Risk Web App is a simple tool to predict the risk of heart disease based on a few parameters of the patient.')
st.caption('This Dashboard is part of the project of the course "Fundamentals of Artificial Intelligence and Knowledge Representation (Mod. 3)" at the Alma Mater Studiorum Università di Bologna. This tool is not meant to be used as a medical diagnosis. Please consult a doctor for a professional opinion. We do not take any responsibility for any decision made based on the output of this tool and the use of this tool is just for educational purposes.')

# with st.expander('About Bayesian Networks')

df = load_dataset()
model = load_model()
probs = np.array([])

LEFT, RIGHT = st.columns(2)

with LEFT:
    left, mid, right = st.columns(3)
    with left:
        age = st.selectbox('Age', df['Age'].unique(),
                           help='Age of the patient. Young: less or equal to 54 | Old: more than 55 years old.')
    with mid:
        sex = st.selectbox('Sex', df['Sex'].unique(),
                           help="The gender of the patient.")

    with right:
        chest_pain_type = st.selectbox(
            'Chest Pain Type', df['ChestPainType'].unique(), index=None, help="The type of chest pain experienced by the patient. TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")

    st.divider()
    left, right = st.columns(2)

    with left:
        resting_bp = st.selectbox('Resting Blood Pressure',
                                  df['RestingBP'].unique(), index=None, help="The resting blood pressure of the patient. Normal: less than 120 | High: between 120 and 140 | Very High: more than 140")
    with right:
        cholesterol = st.selectbox(
            'Cholesterol', df['Cholesterol'].unique(), index=None, help="The cholesterol level of the patient. Optimal: less than 200 | Borderline: between 200 and 239 | High: more than 239")
    with left:
        fasting_bs = st.selectbox(
            'Fasting Blood Sugar', df['FastingBS'].unique(), index=None, help="The fasting blood sugar of the patient. True: if FastingBS > 120 mg/dl, False: otherwise")
    with right:
        max_hr = st.selectbox(
            'Max Heart Rate', df['MaxHR'].unique(), index=None, help="The maximum heart rate of the patient. Low: less than 113 | Medium: between 113 and 157 | High: more than 157")
    with left:
        exercise_angina = st.selectbox(
            'Exercise Angina', df['ExerciseAngina'].unique(), index=None, help="The presence of exercise-induced angina. True: if ExerciseAngina is present, False: otherwise")
    with right:
        oldpeak = st.selectbox('Oldpeak', df['Oldpeak'].unique(
        ), index=None, help="The ST depression induced by exercise relative to rest. Low: less than 2.0 | Medium: between 2.0 and 4.1 | High: more than 4.1")
    with left:
        st_slope = st.selectbox(
            'ST Slope', df['ST_Slope'].unique(), index=None, help="The slope of the peak exercise ST segment. Up: upsloping, Flat: flat, Down: downsloping")

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

        labels = list(query.keys())

        # We would like to know which "exam" should be performed in order to decrease the likelihood of heart disease.
        # We will add one variable at a time as evidence and check the probability of heart disease.

        target = ['HeartDisease']
        variables = []
        evidence = {}

        for label in labels:
            if query[label] == None or query[label] == 'None':
                variables.append(label)
            else:
                evidence[label] = query[label]

        base_result = exact_inference(model, target, evidence)
        probs = base_result.values
        probs = np.round(probs * 100, 2)

        my_dict = {}
        for col in df.drop('HeartDisease', axis=1).columns:
            my_dict[col] = df[col].unique().tolist()

        dummy_df = pd.DataFrame(columns=['exam', 'outcome', 'prob'])

        for var in variables:
            for val in my_dict[var]:
                evidence[var] = str(val)
                result = exact_inference(model, target, evidence)
                dummy_df.loc[len(dummy_df)] = [var, val, result.values[1]]
                del evidence[var]

        dummy_df.sort_values(by='prob', ascending=False,
                             inplace=True)


with RIGHT:
    if probs.shape[0] > 0:
        st.metric(label='Heart Disease Risk (%)',
                  value=probs[1], delta=probs[0], delta_color='inverse')

        st.write(f'Probability of heart disease: {probs[1]} %')

        st.divider()
        if dummy_df.shape[0] > 0:
            st.markdown(
                "The following exams are recommended to find out if the patient has heart disease or not:")
            for i in range(0, min(dummy_df.shape[0], 3)):
                st.markdown(
                    f"{i+1}. **{dummy_df.iloc[i, 0]}** if assessed to **{dummy_df.iloc[i, 1]}** then heart disease probability will be **{dummy_df.iloc[i, 2]:.3f} %**")

st.divider()
with st.expander('Credits & Authors'):
    st.caption(
        'Matteo Fasulo, Luca Tedeschini, Antonio Gravina, Luca Babboni @ 2024 - University of Bologna')
