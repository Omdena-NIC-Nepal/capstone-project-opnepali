
#Import BuiltIn Python Module
import pandas as pd #To handle Data
import seaborn as sns # To create attractive and informative statistical graphics with fewer lines of code than Matplotlib
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt #Plotting library for creating static, animated, or interactive visualizations
import streamlit as st #For StreamLit Pagees

def show():
    st.markdown("## 🧮 Exploratory Data Analysis (EDA)")
    st.markdown("#### 📅 Processed Combined Data Preview")
    
    if 'data' not in st.session_state:
        st.error("Data not loaded. Please return to the Home page.")
        return
    
    _df_data = st.session_state.data
    
    _df_data["year"] = _df_data["year"].astype(str)
    st.dataframe(_df_data.head(), use_container_width=True)
    _df_data["year"] = _df_data["year"].astype(int)

    # Basic overview
    
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Data Shape/Types: Rows: {_df_data.shape[0]}, Columns: {_df_data.shape[1]}")
        st.write(_df_data.dtypes)
    
    with col2:
        st.write("Missing Values")
        st.write(_df_data.isnull().sum())

    st.write("Data Statistics")
    st.write(_df_data.describe())

    # Time series analysis
    st.subheader("Time Series Analysis")
    time_col = st.selectbox("Select a variable for time series plot", _df_data.select_dtypes(include = 'number').columns)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=_df_data, x='year', y=time_col, ax=ax)
    ax.set_title(f"{time_col} over Time")
    st.pyplot(fig)

    # Distribution plots
    st.subheader("Distribution Analysis")
    dist_col = st.selectbox("Select a variable for distribution analysis", _df_data.select_dtypes(include = 'number').columns)

    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    sns.boxplot(data=_df_data, y=dist_col, ax=ax[1])
    sns.histplot(data=_df_data, x=dist_col, kde=True, ax=ax[0])
    st.pyplot(fig)

    # Correlation analysis
    st.subheader("Correlation Analysis")

    fig, ax = plt.subplots(figsize = (10,6))
    sns.heatmap(_df_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Target variable analysis
    st.subheader("Target Variable Analysis")
    st.write("Average Maximum Temperature (avg_max_temp)")

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.boxplot(data=_df_data, y="avg_max_temp", ax=ax[1])
    sns.histplot(data=_df_data, x='avg_max_temp', kde=True, ax=ax[0])
    st.pyplot(fig)

    st.subheader("Top Correlated Features with Target")
    corr_with_target = _df_data.corr()['avg_max_temp'].sort_values(ascending=False)
    st.write(corr_with_target)