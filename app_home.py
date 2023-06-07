import streamlit as st
import pandas as pd


def run_app_home():
    df = pd.read_csv('data/train.csv')
    df1= pd.DataFrame({
    'columns':['Employee ID', 'Date of Joining', 'Gender', 'Company Type',
       'WFH Setup Available', 'Designation', 'Resource Allocation',
       'Mental Fatigue Score', 'Burn Rate'],
    'explain':['직원 ID', '직원이 조직에 가입한 날', '성별 (여성 / 남성)', '회사 종류 (서비스 / 생산)',
       '재택근무 가능 여부 (예 / 아니요)', '직책 (0: 사원,대리/1: 과장/2: 차장/3: 부장/4: 이사,상무/5: 부사장)', '하루 근무시간 ([1.0, 10.0] 범위 / 10.0에 가까울수록 더 많이 일한다)',
       '정신적 피로도 ([0.0, 10.0] 범위 / 10.0에 가까울수록 피로도가 높다)', '번아웃 비율 ([0.0, 1.0] 범위)'],
})  
    st.divider()
    st.markdown('-  이 앱은 직원들의 번아웃 데이터를 이용하여 번아웃이 오는 주 원인을 찾습니다')
    st.markdown('-  직원 정보를 입력하면 데이터를 분석하여 만든 인공지능 모델을 통해 번아웃 비율이 어느정도 되는지 예측합니다')
    st.divider()
    
    if st.button('데이터 보기') : # 버튼을 누르면 True다
        original_title = '<p style="text-align: center; color: grey; font-size : 18px;">직원들의 번아웃 데이터</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        st.dataframe(df)
        st.write('-데이터 크기)')
        st.write(df.shape)
        link='-데이터 출처: [link](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out)'
        st.markdown(link,unsafe_allow_html=True)

    if st.button('데이터 설명') : # 버튼을 누르면 True다
        st.table(df1)
   
