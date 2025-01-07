
import pandas as pd
import os
import joblib
from flask import Flask, request, render_template
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming scaler was saved separately

# WHO thresholds for pollutants
WHO_thresholds = {
    'PM2.5': 25,
    'PM10': 50,
    'NO2': 40,
    'SO2': 20,
    'O3': 100,
    'CO': 4
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            data = pd.read_excel(filepath, engine='openpyxl')
            predictions = predict_air_quality(data)
            return render_template('results.html', tables=[predictions.to_html(classes='data')], titles=predictions.columns.values)
    return render_template('upload.html')


def preprocess_data(data):
    """
    Preprocess the data to match the features used for training the model.
    """
    # Step 1: Extract 'Day', 'Month', 'Year' if 'Date' exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = data['Date'].dt.day
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year

    

    # Step 4: Handle missing values using median imputation
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns  # Ensure numeric columns are selected
    print("Numeric columns detected for imputation:", num_cols)

    # If no numeric columns are detected, raise an error or handle it gracefully
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for imputation!")

    

    # Step 5: Create the target variable 'Exceeds_WHO' based on WHO thresholds
    data['Exceeds_WHO'] = (
    (data['PM2.5'] > WHO_thresholds['PM2.5']) |
    (data['PM10'] > WHO_thresholds['PM10']) |
    (data['NO2'] > WHO_thresholds['NO2']) |
    (data['SO2'] > WHO_thresholds['SO2']) |
    (data['O3'] > WHO_thresholds['O3']) |
    (data['CO'] > WHO_thresholds['CO'])
        ).astype(int)
    # Step 2: Drop unnecessary columns ('City', 'AQI', 'AQI_Bucket') from the dataset
    data = data.drop(columns=['City', 'Date', 'AQI', 'AQI_Bucket','Exceeds_WHO'], errors='ignore')
   # Impute missing values for numerical columns
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    

    # # Step 3: Check the data types
    # print("Data types before imputation:\n", data.dtypes)
    # # Step 6: Ensure all required columns are present (matching the training data)
    # required_columns = [
    #     'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Day', 'Month', 'Year', 'Benzene', 
    #     'NH3', 'NO', 'NOx', 'Toluene', 'Xylene'  # Include all columns used during training
    # ]
    # missing_columns = set(required_columns) - set(data.columns)

    # # Add missing columns with default values (e.g., 0)
    # for col in missing_columns:
    #     data[col] = 0  # Default value, adjust as needed

    # # Step 7: Reorder columns to match the training data
    # data = data[required_columns]  # Ensure column order matches the training data

    # Step 8: Scale the data using the same scaler used during training
    data_scaled = scaler.fit_transform(data)

    return data_scaled


def predict_air_quality(data):
    """
    Function to predict air quality based on the input data.
    """
    # Preprocess the input data to match the model's expected format
    data_scaled = preprocess_data(data)

    # Predict using the loaded model
    predictions = model.predict(data_scaled)

    # Add predictions to the original dataframe
    data['Predicted_AQI_Bucket'] = predictions
    return data


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
