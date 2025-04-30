#Import BuiltIn Python Module
import pandas as pd #To handle Data
import numpy as np #For Numerical Processing
from pathlib import Path #Provides an object-oriented approach to handling filesystem paths.
import matplotlib.pyplot as plt #Plotting library for creating static, animated, or interactive visualizations
import streamlit as st #For StreamLit Pagees
import time #Provides time-related functions — such as sleeping, measuring execution time, or working with timestamps

#- Define a function to read CSV files using pathlib for path handling.
def read_csv(base_path: Path, filename: str):
    """
    Reads a CSV file from the given base_path and filename.
    Args:
        base_path (Path): Directory containing the file.
        filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    file_path = base_path / filename
    return pd.read_csv(file_path)

# Function to cap outliers based on IQR
def cap_outliers(series: pd.Series) -> pd.Series:
    """
    Caps values in a Series at 1.5 * IQR from Q1 and Q3.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return series.clip(lower, upper)


def show():
    st.markdown("## 📝 Data Preprocess")
    st.markdown("#### 📅 Processeing Data and preparing Preview")

    #Load Socio-Economic Data
    st.markdown("")
    st.markdown("###### Loading raw socio-economic for review from *eco-socio-env-health-edu-dev-energy_npl.csv*.")
    #Set socio-economic data directory
    socio_path = Path("data/raw/socio-economic/")
    # Load the main dataset
    _raw_socio_data = read_csv(socio_path, "eco-socio-env-health-edu-dev-energy_npl.csv")
    
    # Display few first rows
    _raw_socio_data["Year"] = _raw_socio_data["Year"].astype(str)
    st.dataframe(_raw_socio_data.head(), use_container_width=True)
    st.success("Done!")

    #============================================================
    #Process Socio-Economic Indicators
    #- Select key indicators for further analysis.
    # Map of indicators to simplified column names
    st.markdown("")
    st.markdown("###### Processing socio-economic data from *eco-socio-env-health-edu-dev-energy_npl.csv* by selecting key indicators and displaying them for review.")
    selected_indicators = {
        'Agricultural land (sq. km)': 'agri_land_area',
        'Permanent cropland (% of land area)': 'cropland_pct',
        'Population density (people per sq. km of land area)': 'population_density',
        'Fertilizer consumption (kilograms per hectare of arable land)': 'fertilizer_kg_per_ha'
    }

    # Filter and reshape data
    filtered = (
        _raw_socio_data[_raw_socio_data['Indicator Name'].isin(selected_indicators.keys())]
        .loc[:, ['Year', 'Indicator Name', 'Value']]
    )

    # Handle duplicate Year-Indicator pairs by averaging
    filtered = (
        filtered
        .groupby(['Indicator Name', 'Year'], as_index=False)
        .agg(Value=('Value', 'mean'))
    )

    # Rename indicators and pivot to wide format
    filtered['Indicator Name'] = filtered['Indicator Name'].map(selected_indicators)
    _processed_socio_data = filtered.pivot(index='Year', columns='Indicator Name', values='Value').reset_index()

    # Impute missing values with median for each column
    for col in _processed_socio_data.columns:
        if col != 'Year':
            _processed_socio_data.fillna({col: _processed_socio_data[col].median()}, inplace=True)

    # Rename Year column to year
    _processed_socio_data.rename(columns={'Year': 'year'}, inplace=True)
    _processed_socio_data["year"] = _processed_socio_data["year"].astype(str)

    # Display processed socio-economic data
    st.dataframe(_processed_socio_data.head(), use_container_width=True)
    st.success("Done!")

    #===================================================
    #Load Climate Data
    #- Read multiple climate-related CSV files and combine into a single DataFrame.  
    st.markdown("")
    st.markdown(
    """
    ###### Reading multiple climate-related CSV files and combining into a single Climate DataFrame. 
    *observed-annual-average-temp.csv*, 
    *observed-annual-average-min-temp.csv*, 
    *observed-annual-average-max-temp.csv*, 
    *observed-annual-relative-humidity.csv*, 
    *observed-annual-average-largest-1-day-precipitation.csv*
    """
    )
    climate_path = Path("data/raw/climate/")

    # Dictionary of climate files and target column names
    climate_files = {
        "observed-annual-average-temp.csv": "avg_mean_temp",
        "observed-annual-average-min-temp.csv": "avg_min_temp",
        "observed-annual-average-max-temp.csv": "avg_max_temp",
        "observed-annual-relative-humidity.csv": "relative_humidity",
        "observed-annual-average-largest-1-day-precipitation.csv": "precipitation_max"
    }

    # Load each file and rename columns
    climate_csv_files = {}
    for fname, col in climate_files.items():
        df = read_csv(climate_path, fname)
        # Assume 'Category' is the year index and 'Annual Mean' is the value
        climate_csv_files[col] = df.set_index('Category')['Annual Mean']

    # Combine into one DataFrame
    _processed_climate_data = pd.DataFrame(climate_csv_files).reset_index().rename(columns={'Category': 'year'})
    _processed_climate_data["year"] = _processed_climate_data["year"].astype(str)

    # Display climate data
    st.dataframe(_processed_climate_data.head(), use_container_width=True)
    st.success("Done!")


    #==================================================
    st.markdown("")
    st.markdown("###### Reading Rainfall data from *npl-rainfall-adm2-full.csv* and merging it with Climate Dataframe.")
    _processed_climate_data["year"] = _processed_climate_data["year"].astype(int)

    #Rainfall Data
    #- Load and aggregate daily rainfall to annual mean.
     
    rainfall = read_csv(climate_path, "npl-rainfall-adm2-full.csv")

    # Drop header row if duplicated and convert types
    rainfall = rainfall.drop(index=0)
    rainfall['date'] = pd.to_datetime(rainfall['date'])
    rainfall['rfh'] = pd.to_numeric(rainfall['rfh'], errors='coerce')

    # Compute annual mean rainfall
    rainfall['year'] = rainfall['date'].dt.year
    annual_rain = rainfall.groupby('year')['rfh'].mean().reset_index(name='annual_rainfall')

    # Merge with _processed_climate_data
    _processed_climate_data = _processed_climate_data.merge(annual_rain, on='year', how='outer')
    _processed_climate_data["year"] = _processed_climate_data["year"].astype(str)

    # Impute missing climate values with column mean
    _processed_climate_data.fillna(_processed_climate_data.mean(numeric_only=True), inplace=True)

    # Display combined climate data
    st.dataframe(_processed_climate_data.head(), use_container_width=True)
    st.success("Done!")


    #==================================================
    st.markdown("")
    st.markdown("###### Merging preprocessed Socio-Economic and Climate Data in a single Climate Dataframe.")
    _processed_climate_data["year"] = _processed_climate_data["year"].astype(int)
    _processed_socio_data["year"] = _processed_socio_data["year"].astype(int)

    #Merge Socio-Economic and Climate Data
    _final_combined_data = pd.merge(_processed_climate_data, _processed_socio_data, on='year', how='outer')
    # Final imputation of any remaining missing values
    _final_combined_data.fillna(_final_combined_data.mean(numeric_only=True), inplace=True)
    _final_combined_data["year"] = _final_combined_data["year"].astype(str)

    st.dataframe(_final_combined_data.head(), use_container_width=True)
    st.success("Done!")

    #=========================================================
    st.markdown("")
    st.markdown("###### Checking final Climate Dataframe for missing values.")
    # Final check for missing values
    st.write(_final_combined_data.isnull().sum())
    st.success("Done!")

    #=========================================================
    #Exploratory Data Analysis: Outlier Detection and Capping
    #-Using IQR method to cap outliers across numerical features.
    st.markdown("")
    st.markdown("###### Capping outliers for all features in final Dataframe and visualizing them before and after capping.")
    _final_combined_data["year"] = _final_combined_data["year"].astype(int)
    num_cols = _final_combined_data.select_dtypes(include=[np.number]).columns.drop('year')

    # Apply capping and visualize before/after
    with st.spinner("Loading..."):
        col1, col2 = st.columns(2)
        
        for col in num_cols:

            fig, ax = plt.subplots(figsize=(10, 2))
            ax.boxplot(_final_combined_data[col].dropna(), vert=False)
            ax.set_title(f"Before capping: {col}")
            with col1:
                st.pyplot(fig)

            # Cap outliers
            _final_combined_data[col] = cap_outliers(_final_combined_data[col])

            fig, ax = plt.subplots(figsize=(10, 2))
            ax.boxplot(_final_combined_data[col].dropna(), vert=False)
            ax.set_title(f"After capping: {col}")
            with col2:
                st.pyplot(fig)
    st.success("Done!")


    #=============================================
    # Save Processed Data
    st.markdown("")
    st.markdown("###### Saving processed data from final Dataframe to disk.")
    output_path = Path("data/processed_data")
    output_path.mkdir(parents=True, exist_ok=True)
    _final_combined_data.to_csv(output_path / "combined_data.csv", index=False)

    st.write("Processed data saved to:", output_path / "combined_data.csv")
    st.success("Done!")

