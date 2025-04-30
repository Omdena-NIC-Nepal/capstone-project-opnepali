#Import BuiltIn Python Module
import streamlit as st #For StreamLit Pagees

# Import custom functions from local python (.py) files
from functions_data_utility import _data_load

def show():
    st.markdown(
    """
    ## ⛅ Climate Data Assessment & Temperature Prediction System for Nepal
    
    ##### Part of assignment under Capstone Project of Omdena's & NIC Capacity Building Batch II-Group B.
    
    ###### The purpose of assignment is to create a data analysis system that monitors, analyzes, and predicts maximum Temperature in Nepal with a focus on vulnerable regions. This application helps **analyze climate data and predict maximum temperatures** based on various environmental and agricultural factors and is developed by Om Nepali.

    #### 🌟 Overview:
    - **Objective**: Predict maximum temperatures for nepal based on historical climate and agricultural data
    - **Target Variable**: Average Maximum Temperature (`avg_max_temp`)
    - **Included Features**: Temperature Metrics, Humidity, Precipitation, Population density, Agricultural Land Area, etc,
    - **Involves**:
        - Exploratory Data Analysis (EDA)
        - Feature Engineering
        - Machine Learning Modeling
        - Natural Language Processing
    """
    )

    st.markdown(
    """
    #### ↪ Navigation Guide:
    - **Home**: About this app and information and Data Sources and their Defnitions.
    - **Data Preprocess**: Preprocesses all the raw data and produce final Climate Data.
    - **Exploratory Data Analysis**: Visualize and understand the data
    - **Feature Engineering**: Create and select important features
    - **Model Training/Evaluation**: Train machine learning modles and Compare model's performance
    - **Temperature Prediction**: Make new predictions with trained models
    - **Basic Text Analysis**: Analyze Basic texts using NLP
    - **Climate Text Analysis**: Analyze Climate texts using NLP
    """
    )

    st.markdown(
    """
    #### 💾 Data Sources and Defnitions:

    ###### **Climate Data**: https://climateknowledgeportal.worldbank.org/country/nepal/era5-historical

    **observed-annual-average-min-temp.csv**			
    - *Included indicators are*:
        - Observed Years (Category)
        - Mean of Min Temperature Observed across the Year (Annual Mean)
    
    **observed-annual-average-max-temp.csv**
    - *Included indicators are*:
        - Observed Years (Category)
        - Mean of Max Temperature Observed across the Year (Annual Mean)

    **observed-annual-average-temp.csv**
    - *Included indicators are*:
        - Observed Years (Category)
        - Mean of Average Temperature Observed across the Year (Annual Mean)

    **observed-annulal-relative-humidity.csv**
    - *Included indicators are*:
        - Observed Years (Category)
        - Mean of Relative Humidity Observed across the Year (Annual Mean)

    **observed-annual-average-largest-1-day-precipitation.csv**
    - *Included indicators are*:
        - Observed Years (Category)
        - Mean of 1 Day largest Precipitation Observed across the Year (Annual Mean)

    ###### **Rainfall Data**: https://data.humdata.org/dataset?q=nepal&sort=last_modified+desc&ext_page_size=25&page=4

    **npl-rainfall-adm2-full.csv**	
    - *Included indicators are (for each dekad)*:
        - 10 day rainfall [mm] (rfh)
        - rainfall 1-month rolling aggregation [mm] (r1h)
        - rainfall 3-month rolling aggregation [mm] (r3h)
        - rainfall long term average [mm] (rfh_avg)
        - rainfall 1-month rolling aggregation long term average [mm] (r1h_avg)
        - rainfall 3-month rolling aggregation long term average [mm] (r3h_avg)
        - rainfall anomaly [%] (rfq)
        - rainfall 1-month anomaly [%] (r1q)
        - rainfall 3-month anomaly [%] (r3q)

    ###### **Socio-economic Data**: https://climateknowledgeportal.worldbank.org/country/nepal/era5-historical
    **eco-socio-env-health-edu-dev-enery-npl.csv**
    - *Included Indicators for Nepal*:
        - Economic, Social, Environmental, Health, Education, Development and Energy indicators.

    Contains data from the World Bank's data portal covering the following topics which also exist as individual datasets on HDX: Agriculture and Rural Development, Aid Effectiveness, Economy and Growth, Education, Energy and Mining, Environment, Financial Sector, Health, Infrastructure, Social Protection and Labor, Poverty, Private Sector, Public Sector, Science and Technology, Social Development, Urban Development, Gender, Millenium development goals, Climate Change, External Debt, Trade.

    ######
    ###### **Sentiment Data**: https://www.kaggle.com/datasets/thedevastator/social-media-sentiment-and-climate-change
    **negative.csv**
    - *Included Indicators*:
        - English Words (Word)
        - Number of Occurrence (frequency)
        - Sentiment of the Word (sentiment)
        - Negative/Positive Sentiment (category)

    **positive.csv**
    - *Included Indicators*:
        - English Words (Word)
        - Number of Occurrence (frequency)
        - Sentiment of the Word (sentiment)
        - Negative/Positive Sentiment (category)
    """
    )


