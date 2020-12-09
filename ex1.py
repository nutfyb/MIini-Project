import streamlit as st
import pandas as pd
import pickle

st.write("""
# Welcome to our web application
""")

st.header('Application of Student\'s Status Prediction:')


st.write("""
### User Input:
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')


def get_input():
    # widgets
    v_AcademicYear = st.sidebar.selectbox('AcademicYear', [ 2562, 2563])
    v_Sex = st.sidebar.radio('Sex', ['Male', 'Female'])   
    v_StudentTH = st.sidebar.radio('StudentTH', ['International students', 'Thai students'])
    v_EntryTypeName = st.sidebar.selectbox('EntryTypeName', ['FOREIGNER','QUOTA 17 NORTHERN PROVINCES','QUOTA BY COMMUNITY HOSPITAL','SPECIAL FOR GOOD STUDENT','GOOD BEHAVE STUDENTS','RE-ID FIRST SEMESTER GPAX 2.00','DIRECT ADMISSION','DIRECT ADMISSION BY SCHOOL','ADMISSIONS','INTERNATIONAL SCHOOL','DISABLE STUDENT','QUOTA BY SCHOOL','SPECIAL TALENT','CHIANG RAI DEVELOPMENT SCHOLARSHIP','DIRECT ADMISSION UNDER CONDITION GPAX 2.00 FIRST SEMESTER (FOREIGN)','EP-MEP PROGRAM','QOUTA FOR SOUTHERN BORDER','SCHOLARSHIP FROM SOUTHERN BORDER'])
    v_TCAS = st.sidebar.selectbox('TCAS', [1, 2, 3, 4, 5])
    v_GPAX = st.sidebar.slider('GPAX', 0.0, 3.00, 4.00)

    if v_Sex == 'Male':
        v_Sex = 'M'
    elif v_Sex == 'Female':
        v_Sex = 'F'

    if v_StudentTH == 'Thai students':
        v_StudentTH = '1'
    elif v_StudentTH == 'International students':
        v_StudentTH = '0'

    # dictionary
    data = {'AcademicYear': v_AcademicYear,
            'Sex': v_Sex,
            'StudentTH': v_StudentTH,           
            'EntryTypeName': v_EntryTypeName,
            'TCAS': v_TCAS,
            'GPAX': v_GPAX, }

    # create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df



data_df = get_input()
st.write(data_df)

data_sample = pd.read_csv('new_sample_tcas.csv')
df = pd.concat([data_df, data_sample], axis=0)


cat_data = pd.get_dummies(df[['Sex','EntryTypeName']])

# cat_data = pd.get_dummies(df[['EntryTypeName']])

# Combine all transformed features together

X_new = pd.concat([df, cat_data], axis=1)
X_new = X_new[:1]
  # Select only the first row (the user input data)
# Drop un-used feature
X_new = X_new.drop(columns=['Sex','EntryTypeName','Sex_Male'])
# X_new = X_new.drop(columns=['EntryTypeName'])


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
st.write("""
### Pre-Processed Input:
""")
st.write(X_new)
# Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write("""
### Normalized Input:
""")
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
X_new = load_knn.predict(X_new)
st.write("""
### Prediction:
""")
st.write(X_new)
