<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=burn_out_employees&fontSize=50" />
- 앱 대시보드: http://43.200.244.94:8502/

1. 앱 설명
- 이 앱은 직원들의 번아웃 데이터를 이용하여 번아웃이 오는 주 원인을 분석하고 직원 정보를 입력하면 그 직원의 번아웃 비율을 예측해줍니다.

2.사용한 데이터
![직원번아웃](https://github.com/YeojinSon7/burn_out_employees/assets/130967465/09b03dba-a1ca-41e9-9c2e-91955706ce1b)
- 출처: https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out
3. 사용한 기술

 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/><img src="https://img.shields.io/badge/Numpy-013243?style=flat&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"/>
 
 <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/><img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white"/>
 
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/Visual Studio Code-007ACC?style=flat&logo=visualstudiocode&logoColor=white"/>
 4. 사용한 머신러닝 모델
 - Linear Regression: LinearRegression()
 - Linear Regression (L2 Regularization): Ridge()
 - Linear Regression (L1 Regularization): Lasso()
 - K-Nearest Neighbors: KNeighborsRegressor()
 - Decision Tree: DecisionTreeRegressor()
 - Random Forest: RandomForestRegressor()
 - Gradient Boosting: GradientBoostingRegressor()
5. 각 모델별 정확도
- Linear Regression R^2 Score: 0.87075
- Linear Regression (L2 Regularization) R^2 Score: 0.87075
- Linear Regression (L1 Regularization) R^2 Score: -0.00001
- K-Nearest Neighbors R^2 Score: 0.85603
- Decision Tree R^2 Score: 0.81653
- Random Forest R^2 Score: 0.89806
- Gradient Boosting R^2 Score: 0.90257
--> 정확도가 가장 높은 Gradient Boosting 모델 이용
6. 사용한 이미지
- 출처: https://greatplacetowork.me/wp-content/uploads/2022/09/mental-employe-mindset-health-progress-UAE-e1662518588611.jpg

