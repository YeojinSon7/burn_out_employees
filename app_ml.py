import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def preprocess_inputs(df):
  df = df.copy()

  df = df.drop('Employee ID',axis=1)

  missing_target_rows = df.loc[df['Burn Rate'].isna(), :].index
  df = df.drop(missing_target_rows, axis=0).reset_index(drop=True)

  for column in ['Resource Allocation', 'Mental Fatigue Score']:
    df[column] = df[column].fillna(df[column].mean())

  df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
  
  # df['Join Year'] = df['Date of Joining'].apply(lambda x: x.year) 2008년 하나밖에 없음
  df['Join Month'] = df['Date of Joining'].apply(lambda x: x.month)
  df['Join Day'] = df['Date of Joining'].apply(lambda x: x.day)
  df = df.drop('Date of Joining', axis=1)

  df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
  df['Company Type'] = df['Company Type'].replace({'Product': 0, 'Service': 1})
  df['WFH Setup Available'] = df['WFH Setup Available'].replace({'No': 0, 'Yes': 1})

  y = df['Burn Rate']
  X = df.drop('Burn Rate', axis=1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
  X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


  return X_train, X_test, y_train, y_test


def run_app_ml():
    df = pd.read_csv('data/train.csv')
    X_train, X_test, y_train, y_test = preprocess_inputs(df)
    st.write('- 7개의 머신러닝 모델들의 검증 결과, 제일 정확도가 높은 모델인 :orange[Gradient Boosting]을 이용하겠습니다')
    if st.checkbox('인공지능 모델 선택과정 보기') == True:
        st.markdown('사용할 인공지능 모델')
        st.text('-  Linear Regression')
        st.text('-  Linear Regression (L2 Regularization)')
        st.text('-  Linear Regression (L1 Regularization)')
        st.text('-  K-Nearest Neighbors')
        st.text('-  Decision Tree')
        st.text('-  Random Forest')
        st.text('-  Gradient Boosting')
        st.divider()
        models = {
        "                     Linear Regression": LinearRegression(),
        " Linear Regression (L2 Regularization)": Ridge(),
        " Linear Regression (L1 Regularization)": Lasso(),
        "                   K-Nearest Neighbors": KNeighborsRegressor(),
        "                         Decision Tree": DecisionTreeRegressor(),
        "                         Random Forest": RandomForestRegressor(),
        "                     Gradient Boosting": GradientBoostingRegressor(),
                                
    }
        for name, model in models.items():
            model.fit(X_train, y_train)
            print(name + " trained.")
        st.markdown('각 모델별 검증')

        for name, model in models.items():
            st.text(name + " R^2 Score: {:.5f}".format(model.score(X_test, y_test)))
    
    st.subheader('직원의 번아웃 비율 예측')

    gender = st.radio('성별 선택',['여자','남자'])
    if gender == '여자' :
        gender = 0
    else:
        gender = 1
    company = st.radio('회사 종류 선택',['생산직','서비스직'])
    if company == '생산직' :
        company = 0
    else:
        company = 1
    home = st.radio('재택 근무 가능여부',['예','아니요'])
    if home == '아니요' :
        home = 0
    else:
        home = 1

    Designation = st.number_input('직책 입력(0.0 ~ 5.0)',0.0,5.0)
    time = st.number_input('근무시간 입력(1.0 ~ 10.0)',1.0,10.0)
    mental = st.number_input('정신적 피로도 입력(0.0 ~ 10.0)',0.0,10.0)
    month = [1,2,3,4,5,6,7,8,9,10,11,12]
    selected_month= st.selectbox('조직에 가입한 날짜(월)',month)
    day = st.number_input('조직한 가입한 날짜(일)',1,31)
    if st.button('번아웃 비율 예측') :
        new_data = np.array([[gender, company, home, Designation, time,mental,selected_month,day]])
        new_data = new_data.reshape(1,8)
        gradient = joblib.load('model/model.pkl')
        y_pred = gradient.predict(new_data)   
        burn = round(y_pred[0],2)
        st.text(str(burn)+'정도의 번아웃 비율이 나왔습니다.')
    
