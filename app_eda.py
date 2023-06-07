import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

 


  return df


def run_app_eda():
    st.subheader('데이터 분석')
    df = pd.read_csv('data/train.csv')
    st.subheader('기본 통계 데이터)')
    st.dataframe(df.describe())

    st.subheader('최대 / 최소 데이터 확인하기)')

    column = st.selectbox('컬럼을 선택하세요.',df.columns[5:])
    st.text('최대 데이터')
    st.dataframe(df.loc[df[column]==df[column].max(),])
    st.text('최소 데이터')
    st.dataframe(df.loc[df[column]==df[column].min(),])

    st.subheader('컬럼 별 Countplot)')

    column = st.selectbox('Countplot를 확인할 컬럼을 선택하세요.',df.columns[2:5])
    fig = plt.figure()
    sns.countplot(x=column,data = df)
    plt.title(column + " Countplot")
    st.pyplot(fig)

    st.subheader('컬럼 별 히스토그램)')

    column = st.selectbox('히스토그램을 확인할 컬럼을 선택하세요.',df.columns[5:])
    bins = st.number_input('빈의 갯수를 입력하세요.',10,30,20) #step부분 디폴트는 1이다
    fig = plt.figure()
    df[column].hist(bins=bins)
    plt.title(column + " Histogram")
    plt.xlabel(column)
    plt.ylabel('count')
    st.pyplot(fig)

    st.subheader('컬럼 별 Boxplot)')

    column = st.selectbox('Boxplot를 확인할 컬럼을 선택하세요.',df.columns[2:5])
    fig = plt.figure()
    sns.boxplot(y= 'Burn Rate',x = column,data = df)
    plt.title(column + " Boxplot")
    plt.xlabel(column)
    plt.ylabel('Burn Rate')
    st.pyplot(fig)


    st.subheader('컬럼 별 산점도)')

    column = st.selectbox('산점도를 확인할 컬럼을 선택하세요.',df.columns[5:8])
    fig = plt.figure()
    plt.scatter(df[column],df['Burn Rate'])
    plt.title(column + " Scatter")
    plt.xlabel(column)
    plt.ylabel('Burn Rate')
    st.pyplot(fig)

    st.subheader('상관 관계 분석)')

    df2= preprocess_inputs(df)
    if st.checkbox('전체 컬럼 상관관계 분석') == True: 
        st.dataframe(df2.corr())
        fig = plt.figure()
        sns.heatmap(data = df2.corr(), annot=True, 
        fmt = '.2f', linewidths=.5, cmap='Blues')
        st.pyplot(fig)
    if st.checkbox('선택 컬럼 상관관계 분석') == True:
        column_list = st.multiselect('상관분석 하고싶은 컬럼을 선택하세요.', df.columns[:])
        if len(column_list) <= 1:
            st.warning('2개 이상 선택하세요')
        else:
            fig2 = plt.figure()
            sns.heatmap(data=df2[column_list].corr(),fmt='.2f',linewidths=0.5, annot = True, vmin = -1, vmax = 1,cmap='coolwarm')
            st.pyplot(fig2)
    st.divider()
    st.subheader('<결론>')
    st.markdown('-  직원들의 번아웃과 상관 관계가 높은 순서대로 나열하자면 :orange[정신적 피로도], :orange[근무시간], :orange[직책]이 있다.')
    st.markdown('-  어쩌면 이 세가지 요소가 번아웃의 주 원인일 수도 있다.')