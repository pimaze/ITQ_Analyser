from flask import Flask, render_template, request
import pandas as pd
import os
import logging
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
VALID_STATUSES = ['Approved', 'Cancelled']
UPDATE_REQUIRED_STATUSES = ['In Review', 'On Hold', 'Ready for Approval', 'Ready for Review', 'Update Required']
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 15 * 1024 * 1024  

# Load analysis criteria from a JSON file
try:
    with open('analysis_criteria.json', 'r') as f:
        ANALYSIS_CRITERIA = json.load(f)
except FileNotFoundError:
    logging.error("analysis_criteria.json not found. Using default criteria.")
    ANALYSIS_CRITERIA = {
        "missing_sla": {"condition": "df['SLA'].isnull()", "message": "Deliverables with missing SLA"},
        "missing_review_cycles": {"condition": "df['# of Review Cycles'].isnull()", "message": "Deliverables with missing '# of Review Cycles'"},
        "additional_review_cycles_missing_reason": {"condition": "(df['# of Review Cycles'] >= 3) & (df['Reason for Additional Review Cycles'].isnull())", "message": "Deliverables with 3+ review cycles and missing 'Reason for Additional Review Cycles'"},
        "invalid_approval_date": {"condition": "(df['Status'] == 'Approved') & (df['Deliverable Approval Date'].isnull())", "message": "Deliverables with 'Approved' status and missing 'Deliverable Approval Date'"},
        "invalid_status": {"condition": "(df['Deliverable Approval Date'].notnull()) & (~df['Status'].isin(['Approved', 'Cancelled']))", "message": "Deliverables with populated 'Deliverable Approval Date' and status different from 'Approved' or 'Cancelled'"},
        "missing_review_received_date": {"condition": "df['Date Deliverable is Received for Review'].isnull()", "message": "Deliverables with missing 'Date Deliverable is Received for Review'"},
        "invalid_approval_date_order": {"condition": "(df['Date Deliverable is Received for Review'].notnull()) & (df['Deliverable Approval Date'].notnull()) & (df['Deliverable Approval Date'] < df['Date Deliverable is Received for Review'])", "message": "Deliverables with 'Deliverable Approval Date' before 'Date Deliverable is Received for Review'"},
        "inconsistent_quick_turnaround_reason": {"condition": "((df['Quick Turnaround Request'].notnull()) & (df['Reason for Quick Turn Around Request'].isnull())) | ((df['Quick Turnaround Request'].isnull()) & (df['Reason for Quick Turn Around Request'].notnull()))", "message": "Deliverables with inconsistent 'Quick Turnaround Request' and 'Reason for Quick Turn Around Request'"},
        "update_required": {"condition": "df['Status'].isin(['In Review', 'On Hold', 'Ready for Approval', 'Ready for Review', 'Update Required']) & (df['Deliverable Approval Date'].isnull())", "message": "Deliverables that require an update"}
    }

# Load fields from a JSON file
try:
    with open('fields_config.json', 'r') as f:
        FIELDS_CONFIG = json.load(f)
        MANDATORY_FIELDS = FIELDS_CONFIG.get('mandatory_fields', ['Deliverable Name', 'Time Spent', 'Deliverable Reviewer'])
        OPTIONAL_FIELDS = FIELDS_CONFIG.get('optional_fields', ['Location of Approved Deliverable'])
except FileNotFoundError:
    logging.error("fields_config.json not found. Using default fields.")
    MANDATORY_FIELDS = ['Deliverable Name', 'Time Spent', 'Deliverable Reviewer']
    OPTIONAL_FIELDS = ['Location of Approved Deliverable']

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df, pod_filter=None, bto_filter=None):
    """Preprocesses the DataFrame: applies filter, removes empty records, converts dates."""
    if pod_filter:
        df = df[df['Related Project::POD'] == pod_filter]

    if bto_filter:
        df = df[df['Related Project::BTO'].isin(bto_filter)]

    df = df.dropna(subset=['Deliverable Name', 'Date Deliverable is Received for Review'], how='all')

    try:
        df['Date Deliverable is Received for Review'] = pd.to_datetime(df['Date Deliverable is Received for Review'], errors='coerce')
        df['Deliverable Approval Date'] = pd.to_datetime(df['Deliverable Approval Date'], errors='coerce')
    except KeyError as e:
        logging.error(f"KeyError during date conversion: . Ensure date columns exist.")
        raise ValueError("Missing required date columns in CSV.") from e
    return df

def apply_analysis_criteria(df):
    """Applies the analysis criteria and returns a dictionary of results."""
    results = {}
    for analysis_key, details in ANALYSIS_CRITERIA.items():
        try:
            condition = eval(details["condition"], {'df': df})
            filtered_df = df[condition]
            deliverable_names = filtered_df['Deliverable Name'].tolist()
            results[analysis_key] = {"message": details["message"], "deliverables": deliverable_names}
        except Exception as e:
            logging.error(f"Error evaluating condition for : ")
            results[analysis_key] = {"message": f"Error evaluating condition: {details['message']}", "deliverables": []}
    return results

def check_missing_fields(df, fields, message):
    """Checks for missing fields in the DataFrame and returns a dictionary of results."""
    missing_fields_df = df[df[fields].isnull().any(axis=1)]
    missing_fields_results = {}
    for index, row in missing_fields_df.iterrows():
        missing_fields = [field for field in fields if pd.isnull(row[field])]
        deliverable_name = row['Deliverable Name']
        missing_fields_results[deliverable_name] = missing_fields
    return {"message": message, "deliverables": missing_fields_results}

def analyze_deliverables(filepath, pod_filter=None, bto_filter=None):
    """Analyzes a CSV file containing deliverable data and identifies various issues."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: ")
        return {"error": "File not found"}
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: . ")
        return {"error": "Invalid CSV file format"}
    except Exception as e:
        logging.exception(f"Unexpected error during CSV reading: ")
        return {"error": "An unexpected error occurred while reading the CSV file."}

    try:
        df = preprocess_data(df, pod_filter, bto_filter)
    except ValueError as e:
        logging.error(f"Error during preprocessing: ")
        return {"error": str(e)}

    if df.empty:
        logging.warning("DataFrame is empty after preprocessing.")
        return {"warning": "No data to analyze after preprocessing."}

    results = apply_analysis_criteria(df)
    results["missing_mandatory_fields"] = check_missing_fields(df, MANDATORY_FIELDS, "Deliverables with missing mandatory fields")
    results["missing_optional_fields"] = check_missing_fields(df, OPTIONAL_FIELDS, "Deliverables with missing optional fields")
    return results


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    pod_filter = request.form.get('pod_filter', '').strip()
    bto_filter = request.form.getlist('bto_filter')

    if file.filename == '':
        return "No selected file"

    if not allowed_file(file.filename):
        return "Invalid file type. Only CSV files are allowed."

    if len(file.read()) > MAX_FILE_SIZE:
        return f"File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024)} MB."
    file.seek(0)  # Reset file pointer after reading

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, filename)
            file.save(file_path)
            analysis_results = analyze_deliverables(file_path, pod_filter, bto_filter)
            os.remove(file_path)  # remove file from disk

            return render_template('results.html', analysis_results=analysis_results)
        except FileNotFoundError:
            return "File not found"
        except pd.errors.ParserError:
            return "Invalid CSV file format"
        except Exception as e:
            logging.exception("Error processing file:")
            return f"Error processing file: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)