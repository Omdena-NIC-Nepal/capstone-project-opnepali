# Project Documentation for Climate Data Assessment & Temperature Prediction System for Nepal

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Installation](#installation)
4. [Executing the Application](#executing-the-application)
5. [Features](#features)

## Overview

The purpose of assignment is to create a data analysis system that monitors, analyzes, and predicts maximum Temperature in Nepal with a focus on vulnerable regions. This application helps analyze climate data and predict maximum temperatures based on various environmental and agricultural factors. The application includes:

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- ML based Model Training & Evaluation
- Temperature Prediction
- NLP based Basic Text Analysis
- NLP based Climate Text Analysis

**Target Variable**: Average Maximum Temperature (`avg_max_temp`)

## File Structure

```
capstone-project-opnepali/
│
 data/                # Main application file
│   ├── processed_data/       # processed final data
│   │   └── combined_data.csv       # combination of all the raw data with selected features and data │ng
│   ├── raw/       # raw data
│   │   ├── climate       # raw climate data
│   │   │   ├── npl-rainfall-adm2-full.csv       # rainfall data
│   │   │   ├── observed_annual-average-largest-1-day-precipitation.csv       # precipitation data
│   │   │   ├── observed-annual-average_temp.csv       # annual average mean temperature data
│   │   │   ├── observed-annual-average-max-temp.csv       # annual average maximum temperature data
│   │   │   ├── observed-annual-average-min-temp.csv       # annual average minimum temperature data
│   │   │   └── observed-annual-relative-humidity.csv       # annual relative humidity data
│   │   └── socio-economic       # raw socio-economic data
│   │       └── eco-socio-env-health-edu-dev-energy_npl.csv       # socio-economic data
│   └── sentiment_data/          # sentimental data on climate change for NLP
│       └── positive.csv         # positive sentiment dataset
│       └── negative.csv         # negative sentiment dataset
├── models/       #Model Files (Files after Pickling and Unpickling)
│   ├── gradient_boosting.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── ridge_regression.pkl
├── utils/        #python Files (Individual file for classified processes)
│   ├── _Home.py                       # Project introduction
│   ├── _Data_Preporocess_.py          # Preprocess, merge and combine raw data to final data.
│   ├── _EDA.py                        # Exploratory Data Analysis
│   ├── _Feature_Engineering.py        # Feature engineering
│   ├── _Model_TrainingEvaluation.py   # Model Training & Evaluation
│   ├── _Prediction.py                 # Max Temperature Prediction
│   ├── _Basic_Text_Analysis.py        # NLP based Basic Text Analysis
│   └── _Climate_Text_Analysis.py      # NLP based Climate Text Analysis
├── app.py                # Main application file
├── functions_data_utility.py          # Utility functions for data processing
├── functions_nltj_utility.py          # Utility functions for NPL processing
├── requirements.txt                   # Dependencies
├── README.txt                         # Project instruction
└── Documentation.md                   # Project Documentation
```

## Installation

### Prerequisites
- Python 3.8+
- Dependecies as defined requirements.txt

### Setup Instructions
1. Clone the repository:
   ```cmd
   git clone https://github.com/Omdena-NIC-Nepal/capstone-project-opnepali.git
   cd capstone-project-opnepali
   ```

2. Create and activate a virtual environment (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

4. Download the spaCy language model:
   ```cmd
   python -m spacy download en_core_web_sm
   ```

5. Place the `combined_data.csv` file in the directory: `../data/processed_data'

## Executing the Application
Start the Streamlit application:
```cmd
streamlit run app.py
```
The application will open in default browser at `http://localhost:8501`

## Features

### Base Features
- **Interactive Data Exploration**: Visualize trends, distributions, and correlations
- **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression, Ridge Regression
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Feature Engineering**: Create and select features for modeling
- **Prediction Interface**: Make new predictions with trained models
- **NLP Integration**: Analyze climate-related text using spaCy

### Technical Highlights
- Modular architecture with separate pages
- Session state management for data persistence
- Model saving functionality
- Responsive UI with interactive components

---