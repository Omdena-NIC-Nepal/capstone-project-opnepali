#Import BuiltIn Python Module
import pandas as pd #To handle Data
import numpy as np #For Numerical Processing
import seaborn as sns # To create attractive and informative statistical graphics with fewer lines of code than Matplotlib
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt #Plotting library for creating static, animated, or interactive visualizations
import streamlit as st #For StreamLit Pagees

#Import modules from Scikit-learn that provides: Data preprocessing, Model training, Model evaluation etc.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Import custom functions from local python (.py) files
from functions_data_utility import _data_preprocess, _model_save

def show():
    st.markdown("## 🤖 Model Training")
    if 'data' not in st.session_state:
        st.error("Data not loaded. Please return to the Home page.")
        return
    
    _df_data = st.session_state.data

    # Model selection
    st.markdown("### 1. Select Model")
    model_options = {
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge  
    }

    selected_model = st.selectbox("Choose a model to train", list(model_options.keys()))

    # Hyperparameter tuning
    st.markdown("### 2. Hyperparameters")
    params = {}

    if selected_model == "Random Forest":
        params['n_estimators'] = st.slider("Number of tress", 10, 200, 100)
        params['max_depth'] = st.slider("Max depth", 1, 20, 10)
        params['random_state'] = 42

    elif selected_model == "Gradient Boosting":
        params['n_estimators'] = st.slider("Number of tress", 10, 200, 100)
        params['learning_rate'] = st.slider("Learning rate", 0.01, 1.0, 0.1)
        params['max_depth'] = st.slider('Max depth', 1, 10, 3)
        params['random_state'] = 42

    elif selected_model == "Ridge Regression":
        params['alpha'] = st.slider("Alpha (Regularization strength)", 0.01, 10.0, 1.0)
        params['solver'] = st.selectbox("Solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        params['random_state'] = 42

    # Train/test split
    st.markdown("### 3. Train/Test Split")
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = _data_preprocess(
        _df_data, test_size=test_size
    )

    # Train model
    if st.button("Train Model"):
        st.markdown("")
        st.markdown("### Training Results")
        with st.spinner(f"Training {selected_model}..."):
            # Initialize model
            model_class = model_options[selected_model]
            model = model_class(**params)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.success("Model trained successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MSE", f"{mse:.2f}")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R² Score", f"{r2:.2f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.markdown("### Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)
            
            # Save model
            _model_save(model, selected_model.lower().replace(" ", "_"))
            st.session_state.trained_model = model
            st.session_state.model_name = selected_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred

            st.success("Model saved successfully! You can now evaluate it in the Model Evaluation page.")

            #=================================
            st.markdown("")
            st.markdown("---")
            st.markdown("## 📈 Model Evaluation")

            # Check if model is trained
            if 'trained_model' not in st.session_state:
                st.warning("No trained model found. Please train a model first.")
                return

            model = st.session_state.trained_model
            model_name = st.session_state.model_name
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred

            st.markdown(f"### Evaluating {model_name} Model")

            # Metrics
            st.markdown("#### 1. Performance Metrics")

            #from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MSE", f"{mse:.2f}")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R² Score", f"{r2:.2f}")

            # Actual vs Predicted plot
            st.markdown("")
            st.markdown("#### 2. Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize = (10,6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)

            # Residual plot
            st.markdown("#### 3. Residual Analysis")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(1, 2, figsize=(15,5))
            sns.histplot(residuals, kde=True, ax=ax[0])
            ax[0].set_title("Residual Distribution")
            sns.scatterplot(x=y_pred, y=residuals, ax=ax[1])
            ax[1].axhline(y=0, color='r', linestyle='--')
            ax[1].set_title("Residuals vs Predicted")
            st.pyplot(fig)

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 4. Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)

            # Save evaluation results
            if st.button("Save Evaluation Results"):
                evaluation_results = {
                    'model': model_name,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                }
                st.session_state.evaluation_results = evaluation_results
                st.success("Evaluation results saved!")