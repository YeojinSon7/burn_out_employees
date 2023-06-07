import streamlit as st
import pandas as pd
from app_home import run_app_home
from app_eda import run_app_eda
from app_ml import run_app_ml
from streamlit_option_menu import option_menu


def main() :
    st.subheader('꿩먹고 알먹고?')
    st.title('정신건강 얻고 효율 얻고!')

    st.sidebar.image("https://greatplacetowork.me/wp-content/uploads/2022/09/mental-employe-mindset-health-progress-UAE-e1662518588611.jpg")
    with st.sidebar:
        selected = option_menu("Manual", ["Home", 'EDA','ML'], 
        icons=['house', 'graph-up','person-gear'], menu_icon="cast", default_index=1)
    
    if selected == "Home":
        run_app_home()
    elif selected == "EDA":
        run_app_eda()
    elif selected == "ML":
        run_app_ml()


if __name__ == '__main__':
    main()