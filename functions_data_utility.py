#Import BuiltIn Python Module
import os #To work with Directory and Files
import pandas as pd #To handle Data
import numpy as np #For Numerical Processing
from sklearn.model_selection import train_test_split #To split data in Train/Test
from sklearn.preprocessing import StandardScaler #To standardizes Features
import joblib #To save Trained Models in ML

#Function to Load the climate data.
def _data_load():
    # Load the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'data', 'processed_data', 'combined_data.csv')
    _df_data = pd.read_csv(data_file)

    # Convert the 'year' column to the integer type (int)
    _df_data['year'] = _df_data['year'].astype(int)

    return _df_data

#Function to Preprocess data for modeling.
def _data_preprocess(_df_data, target_col = 'avg_max_temp', test_size=0.2, random_state=42):
    # Seperate features and target
    X = _df_data.drop(columns = [target_col])
    y = _df_data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=np.number).columns
    print(num_cols)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler

#Function to save Trained Model to disk.
def _model_save(model, model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'models', model_name)
    joblib.dump(model, f"{data_file}.pkl") #Pickling trained machine learning models. Converting a Python object into a byte stream.

#Function to load Trained Model to disk.
def _model_load(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'models', model_name)
    try:
        return joblib.load(f"{data_file}.pkl")
    except AttributeError:
        return joblib.load(f"models/{data_file}.pkl")  #Unpickling trained machine learning models. Reading back byte stream and restoring it to its original Python object.

