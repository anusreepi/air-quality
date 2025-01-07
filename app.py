
import datetime
import pandas as pd
import os
import joblib
from flask import Flask, request, render_template,send_file, url_for
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt

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
            # Generate PDF report
            pdf_filename = os.path.join('uploads', 'air_quality_report.pdf')
            generate_pdf(predictions, pdf_filename)
            # return render_template('results.html', tables=[predictions.to_html(classes='data')], titles=predictions.columns.values)
            # Render predictions and provide PDF download link
            return render_template(
                'results.html',
                tables=[predictions.to_html(classes='data')],
                titles=predictions.columns.values,
                pdf_url=url_for('download_file', filename='air_quality_report.pdf')
            )
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
def categorize_aqi(predictions):
    """
    Categorize AQI values into predefined buckets.
    """
    categories = []
    for value in predictions:
        if value <= 50:
            categories.append('Good')
        elif 51 <= value <= 100:
            categories.append('Moderate')
        else:
            categories.append('Unhealthy')
    return categories

def predict_air_quality(data):
    """
    Function to predict air quality based on the input data.
    """
    # Preprocess the input data to match the model's expected format
    data_scaled = preprocess_data(data)
    # Predict using the loaded model
    predictions = model.predict(data_scaled)
    # Add predictions to the original dataframe
    # data['Predicted_AQI_Bucket'] = predictions
    data['Predicted_AQI_Bucket'] = categorize_aqi(predictions)
    return data

def generate_pdf(data, filename):
    """
    Generate a summarized PDF report for air quality predictions.
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, 'Air Quality Prediction Summary Report', border=False, ln=True, align='C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', border=False, align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # General summary
    total_records = len(data)
    exceeds_who_count = data['Exceeds_WHO'].sum()
    
    pdf.cell(0, 10, f'Total records processed: {total_records}', ln=True)
    pdf.cell(0, 10, f'Records exceeding WHO thresholds: {exceeds_who_count}', ln=True)
    pdf.ln(10)

    # Pollutant-specific summary
    pdf.cell(0, 10, 'Pollutant Summary (Average, Min, Max):', ln=True)
    for pollutant in ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']:
        avg_value = data[pollutant].mean()
        min_value = data[pollutant].min()
        max_value = data[pollutant].max()
        pdf.cell(0, 10, f'  {pollutant}: Avg={avg_value:.2f}, Min={min_value}, Max={max_value}', ln=True)
    pdf.ln(10)

    # Predicted AQI buckets summary
    pdf.cell(0, 10, 'Predicted AQI Buckets:', ln=True)
    aqi_counts = data['Predicted_AQI_Bucket'].value_counts()
    for bucket, count in aqi_counts.items():
        pdf.cell(0, 10, f'  {bucket}: {count}', ln=True)

    # Save the PDF to the specified filename
    pdf.output(filename)

@app.route('/download/<filename>')
def download_file(filename):
    """
    Route to download the generated PDF file.
    """
    filepath = os.path.join('uploads', filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
