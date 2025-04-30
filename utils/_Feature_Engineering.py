#Import BuiltIn Python Module
import pandas as pd #To handle Data
import numpy as np #For Numerical Processing
import streamlit as st #For StreamLit Pagees

def show():
    st.markdown("## 🛠️ Feature Engineering")
    if 'data' not in st.session_state:
        st.error("Data not loaded. Please return to the Home page.")
        return
    
    _df_data = st.session_state.data.copy()
    
    st.markdown("#### Existing Features")
    st.write(_df_data.columns.tolist())
    st.markdown("#### Add New Features")

    # Feature creation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Temperature Features")
        if st.checkbox("Add Temperature Range (max - min)"):
            _df_data['temp_range'] = _df_data['avg_max_temp'] - _df_data['avg_min_temp']
        
        if st.checkbox("Add Temperature Anomaly (deviation from mean)"):
            mean_temp = _df_data['avg_mean_temp'].mean()
            _df_data['temp_anomaly'] = _df_data['avg_mean_temp'] - mean_temp
    
    with col2:
        st.markdown("##### Time Features")
        if st.checkbox("Add Decade Feature"):
            _df_data['decade'] = (_df_data['year'] // 10) * 10
        
        if st.checkbox("Add Year Difference from Reference (2000)"):
            _df_data['years_from_2000'] = _df_data['year'] - 2000
    
    st.markdown("##### Feature Selection")
    st.write("Select features to include in the model:")
    
    # Let user select features
    all_features = [col for col in _df_data.columns if col != 'avg_max_temp']
    selected_features = st.multiselect(
        "Choose features", 
        all_features,
        default=all_features
    )
    
    # Add target back
    selected_features_with_target = selected_features + ['avg_max_temp']
    _engineered_data = _df_data[selected_features_with_target]
    
    # Save to session state
    if st.button("Apply Feature Engineering"):
        st.session_state.data = _engineered_data
        st.success("Feature engineering applied! The dataset now has:")
        st.write(f"{_engineered_data.shape[1]} features, {_engineered_data.shape[0]} rows")
        
        st.markdown("##### Preview of Engineered Data")
        _engineered_data["year"] = _engineered_data["year"].astype(str) #Let Streamlit display the Year field in dataframe without comma.
        st.dataframe(_engineered_data.head(), use_container_width=True)
        _engineered_data["year"] = _engineered_data["year"].astype(int) #Streamlit has displayed the dataframe. Setting Year field back to original Int.
    
    st.write("**To review the Feature Importance Analysis, please proceed to 'Model Training'. After training a model, you can view feature importance right down in the same page**.", unsafe_allow_html=True)