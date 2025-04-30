#Import BuiltIn Python Module
import pandas as pd #To handle Data
import numpy as np #For Numerical Processing
import matplotlib.pyplot as plt #Plotting library for creating static, animated, or interactive visualizations
import seaborn as sns # To create attractive and informative statistical graphics with fewer lines of code than Matplotlib
import streamlit as st #For StreamLit Pagees

# Import custom functions from local python (.py) files
from functions_data_utility import _model_load, _data_preprocess


def show():
    st.markdown("## ☀️ Temperature Prediction")

    models = {}
    model_files = {
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting",
            "Linear Regression": "linear_regression",
            "Ridge Regression": "ridge_regression"
            }

    # Load trained models
    for name, file in model_files.items():
        try:
            models[name] = _model_load(file)
        except FileNotFoundError:
            st.warning(f"Model not found: {name} (looking for {file})")
           
    if not models:
        st.error("No models found. Please train the model from Model Training page.")
        return
    
    # Model selection
    selected_model = st.selectbox("Select a trained model", list(models.keys()))
    model = models[selected_model]

    st.markdown(f"### Make Prediction with {selected_model}")

    # Get feature names (excluding year and target)
    if 'data' not in st.session_state:
        st.error("Data not loaded. Please return to the Home Page.")
        return
    
    _df_data = st.session_state.data
    _df_data["year"] = _df_data["year"].astype(int)

    feature_cols = [col for col in _df_data.columns if col not in ['year', 'avg_max_temp']]

    # Create input form
    st.markdown("#### Input Features")
    input_data = {}

    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        with cols[i%3]:
            if _df_data[col].dtype in ['int64', 'float64']:
                min_val = float(_df_data[col].min())
                max_val = float(_df_data[col].max())
                default_val = float(_df_data[col].median())
                input_data[col] = st.number_input(
                    col, min_value=min_val, max_value=max_val, value=default_val
                )
            else: 
                input_data[col] = st.selectbox(col, _df_data[col].unique())
    
    # Year input
    year = st.number_input("Year", min_value=1950, max_value=2050, value=2025)
    input_data['year'] = year


    # Make prediction
    if st.button("Predict Maximum Temperature"):
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        # Remove last column and store it
        last_col = input_df.pop(input_df.columns[-1])

        # Insert it at position 0
        input_df.insert(0, last_col.name, last_col)
        print(input_df)
        
        # Preprocess (same as training)
        _, _, _, _, scaler = _data_preprocess(_df_data)
        
        # Scale numerical features
        num_cols = input_df.select_dtypes(include=np.number).columns
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Predict
        try:
            prediction = model.predict(input_df.drop(columns=['avg_max_temp'], errors='ignore'))
            # Display result
            st.success(f"Predicted Maximum Temperature: {prediction[0]:.2f}°C")
            
            # Show historical context
            st.markdown("#### Historical Context")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=_df_data, x='year', y='avg_max_temp', ax=ax)
            ax.axvline(x=year, color='r', linestyle='--', label='Prediction Year')
            ax.axhline(y=prediction[0], color='g', linestyle='--', label='Predicted Temp')
            ax.set_title("Historical Maximum Temperatures")
            ax.legend()
            st.pyplot(fig)
        except ValueError:
            st.warning(f"Train the selected model before predicting!")
        