# analyzer.py

import logging
import json
import pandas as pd

# Configure logging (can be configured globally in app.py, but basic setup here is fine)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Configuration ---
def load_json_config(filepath, default_value):
    """Loads configuration from a JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"{filepath} not found. Using default configuration.")
        return default_value
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {filepath}. Using default configuration.")
        return default_value
    except Exception as e:
        logging.error(f"Unexpected error loading {filepath}: {e}. Using default configuration.")
        return default_value

# Load configurations at module level when analyzer is imported
ANALYSIS_CRITERIA = load_json_config('analysis_criteria.json', {
    # Sensible defaults if file is missing/corrupt
    "missing_sla": {"condition": "df['SLA'].isnull()", "message": "Deliverables with missing SLA", "severity": "Medium"},
    "missing_review_cycles": {"condition": "df['# of Review Cycles'].isnull()", "message": "Deliverables with missing '# of Review Cycles'", "severity": "Low"},
    "invalid_approval_date": {"condition": "(df['Status'] == 'Approved') & (df['Deliverable Approval Date'].isnull())", "message": "'Approved' status but missing 'Deliverable Approval Date'", "severity": "High"},
    "invalid_status": {"condition": "(df['Deliverable Approval Date'].notnull()) & (~df['Status'].isin(['Approved', 'Cancelled']))", "message": "Approval date present but status not 'Approved' or 'Cancelled'", "severity": "High"},
    # Add other defaults as needed
})

FIELDS_CONFIG = load_json_config('fields_config.json', {
    "mandatory_fields": ['Deliverable Name', 'Time Spent', 'Deliverable Reviewer'],
    "optional_fields": ['Location of Approved Deliverable']
})
MANDATORY_FIELDS = FIELDS_CONFIG.get('mandatory_fields', [])
OPTIONAL_FIELDS = FIELDS_CONFIG.get('optional_fields', [])

# --- Analysis Helper Functions ---
def preprocess_data(df, pod_filter=None, bto_filter=None):
    """Preprocesses the DataFrame: applies filter, removes empty records, converts dates."""
    original_count = len(df)
    logging.info(f"Preprocessing data. Initial row count: {original_count}")

    if pod_filter:
        try:
            if 'Related Project::POD' not in df.columns:
                 raise ValueError("Column 'Related Project::POD' not found in CSV for filtering.")
            df = df[df['Related Project::POD'] == pod_filter]
            logging.info(f"Applied POD filter '{pod_filter}'. Row count: {len(df)}")
        except KeyError: # Should be caught by the check above, but keep for safety
            raise ValueError("Column 'Related Project::POD' not found in CSV for filtering.")

    if bto_filter:
        try:
            if 'Related Project::BTO' not in df.columns:
                 raise ValueError("Column 'Related Project::BTO' not found in CSV for filtering.")
            df = df[df['Related Project::BTO'].isin(bto_filter)]
            logging.info(f"Applied BTO filter '{bto_filter}'. Row count: {len(df)}")
        except KeyError:
            raise ValueError("Column 'Related Project::BTO' not found in CSV for filtering.")

    # Define essential columns that should exist for basic processing/analysis
    required_columns = {'Deliverable Name', 'Date Deliverable is Received for Review', 'Deliverable Approval Date', 'Status', '# of Review Cycles', 'Reason for Additional Review Cycles', 'Quick Turnaround Request', 'Reason for Quick Turn Around Request', 'SLA'}
    # Check only for columns actually used in default criteria + mandatory/optional fields
    check_cols = required_columns.union(set(MANDATORY_FIELDS)).union(set(OPTIONAL_FIELDS))

    if not check_cols.issubset(df.columns):
        missing = check_cols - set(df.columns)
        # Only raise error if critical columns used in multiple checks are missing
        critical_missing = {'Deliverable Name', 'Status'}.intersection(missing)
        if critical_missing:
             raise ValueError(f"Missing critical columns required for analysis: {', '.join(critical_missing)}")
        else:
            logging.warning(f"Missing columns (may affect some checks): {', '.join(missing)}")

    # Drop rows where both essential identifiers are missing
    # Ensure 'Deliverable Name' exists before using it here
    if 'Deliverable Name' in df.columns and 'Date Deliverable is Received for Review' in df.columns:
        df = df.dropna(subset=['Deliverable Name', 'Date Deliverable is Received for Review'], how='all')
        logging.info(f"Dropped rows with missing essential identifiers. Row count: {len(df)}")

    # Convert date columns safely if they exist
    date_cols = ['Date Deliverable is Received for Review', 'Deliverable Approval Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logging.info(f"Converted date column: {col}")
        else:
             logging.warning(f"Date column '{col}' not found for conversion.")

    if len(df) == 0 and original_count > 0:
         logging.warning("No data remaining after applying filters.")

    return df

def apply_analysis_criteria(df):
    """Applies the analysis criteria and returns a dictionary of results."""
    results = {}
    if df.empty:
        logging.warning("DataFrame is empty, skipping analysis criteria application.")
        return results

    # Ensure 'Deliverable Name' exists for reporting issues
    if 'Deliverable Name' not in df.columns:
        logging.error("Cannot apply criteria: 'Deliverable Name' column is missing.")
        return {"error": {"message": "'Deliverable Name' column missing", "deliverables": []}}

    for analysis_key, details in ANALYSIS_CRITERIA.items():
        condition_str = details.get("condition")
        message = details.get("message", f"Issues for {analysis_key}")
        severity = details.get("severity", "Medium") # Get severity

        if not condition_str:
            logging.warning(f"Skipping criteria '{analysis_key}' due to missing 'condition'.")
            continue

        try:
            # Basic check for required columns within the condition string
            # This is imperfect but better than nothing before eval
            required_cols_for_eval = [col.strip("'[]") for col in df.columns if f"df['{col}']" in condition_str or f'df["{col}"]' in condition_str]

            if not all(col in df.columns for col in required_cols_for_eval):
                 missing_in_cond = set(required_cols_for_eval) - set(df.columns)
                 logging.warning(f"Skipping criteria '{analysis_key}' due to missing columns needed for condition: {missing_in_cond}")
                 results[analysis_key] = {"message": message, "severity": severity, "deliverables": [f"Skipped - Missing Columns: {', '.join(missing_in_cond)}"]}
                 continue

            # Evaluate the condition safely
            condition = eval(condition_str, {'df': df, 'pd': pd})
            filtered_df = df.loc[condition] # Use .loc for boolean indexing
            deliverable_names = filtered_df['Deliverable Name'].tolist()
            results[analysis_key] = {"message": message, "severity": severity, "deliverables": deliverable_names}

        except SyntaxError as e:
            logging.error(f"Syntax error in condition for '{analysis_key}': {condition_str}. Error: {e}")
            results[analysis_key] = {"message": message, "severity": "High", "deliverables": ["Syntax Error in Criteria"]}
        except KeyError as e:
             logging.error(f"Missing column {e} required for criteria '{analysis_key}': {condition_str}")
             results[analysis_key] = {"message": message, "severity": "High", "deliverables": [f"Missing Column Error: {e}"]}
        except Exception as e:
            logging.exception(f"Error evaluating condition for '{analysis_key}': {condition_str}. Error: {e}")
            results[analysis_key] = {"message": message, "severity": "High", "deliverables": ["Evaluation Error"]}
    return results

def check_missing_fields(df, fields, message, severity="Medium"):
    """Checks for missing fields, returns results dict including severity."""
    missing_fields_results = {}
    if df.empty:
        logging.warning(f"DataFrame is empty, skipping missing field check for: {message}")
        return {"message": message, "severity": severity, "deliverables": missing_fields_results}

    # Ensure Deliverable Name exists
    if 'Deliverable Name' not in df.columns:
         logging.error(f"Cannot check missing fields for '{message}': 'Deliverable Name' column missing.")
         # Return an indication of the problem
         return {"message": message, "severity": "High", "deliverables": {"Error": "'Deliverable Name' column missing"}}

    # Ensure all fields to check actually exist in the DataFrame
    fields_to_check = [f for f in fields if f in df.columns]
    missing_cols = set(fields) - set(fields_to_check)
    if missing_cols:
        logging.warning(f"Columns missing from input for '{message}' check: {', '.join(missing_cols)}")

    if not fields_to_check:
        logging.warning(f"No valid columns found for missing field check: {message}")
        return {"message": message, "severity": severity, "deliverables": missing_fields_results}

    try:
        # Check for NaNs or Nulls in the relevant subset of columns
        # Important: Use .loc to avoid SettingWithCopyWarning if df is a slice
        missing_indices = df.loc[:, fields_to_check].isnull().any(axis=1)
        missing_fields_df = df.loc[missing_indices]

        for index, row in missing_fields_df.iterrows():
            missing = [field for field in fields_to_check if pd.isnull(row[field])]
            deliverable_name = row.get('Deliverable Name') # Already checked if column exists
            if deliverable_name and missing:
                missing_fields_results[deliverable_name] = missing
    except Exception as e:
        logging.exception(f"Error during missing field check for '{message}'. Error: {e}")
        # Indicate the error in the results for this check
        return {"message": message, "severity": "High", "deliverables": {"Error": "An error occurred during check"}}

    return {"message": message, "severity": severity, "deliverables": missing_fields_results}


def analyze_deliverables(filepath, pod_filter=None, bto_filter=None):
    """Main analysis function. Reads CSV, preprocesses, runs checks."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logging.info(f"Successfully read CSV: {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return {"error": "Uploaded file not found during analysis."} # User-friendly error
    except pd.errors.EmptyDataError:
        logging.warning(f"CSV file is empty: {filepath}")
        return {"error": "The uploaded CSV file is empty."}
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {filepath}. Error: {e}")
        return {"error": "Invalid CSV file format. Please check the structure and encoding."}
    except Exception as e:
        logging.exception(f"Unexpected error reading CSV: {filepath}")
        return {"error": "An server error occurred while reading the CSV file."}

    try:
        df_processed = preprocess_data(df.copy(), pod_filter, bto_filter)
    except ValueError as e: # Catch specific preprocessing errors (e.g., missing columns)
        logging.error(f"Preprocessing error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.exception("Unexpected error during preprocessing.")
        return {"error": "An server error occurred during data preprocessing."}

    if df_processed.empty:
        logging.warning("DataFrame is empty after preprocessing/filtering. No analysis performed.")
        # Return structure consistent with successful analysis but indicate no data
        return {
             "info": "No data matching the selected filters was found.",
             "missing_mandatory_fields": {"message": "Deliverables with missing mandatory fields", "severity": "High", "deliverables": {}},
             "missing_optional_fields": {"message": "Deliverables with missing optional fields", "severity": "Low", "deliverables": {}},
             # Add other keys with empty deliverables if needed
         }

    # Apply core analysis criteria
    results = apply_analysis_criteria(df_processed)

    # Check missing fields (pass severity)
    results["missing_mandatory_fields"] = check_missing_fields(df_processed, MANDATORY_FIELDS, "Deliverables with missing mandatory fields", severity="High")
    results["missing_optional_fields"] = check_missing_fields(df_processed, OPTIONAL_FIELDS, "Deliverables with missing optional fields", severity="Low")

    logging.info("Analysis complete.")

    # --- Calculate Summary Statistics --- #
    total_deliverables = len(df_processed)
    total_issues = 0
    issues_by_category = {}

    for key, result in results.items():
        if key not in ['error', 'info'] and isinstance(result, dict) and 'deliverables' in result:
            num_issues = 0
            if isinstance(result['deliverables'], dict):
                num_issues = len(result['deliverables'])  # missing field checks have a dictionary
            elif isinstance(result['deliverables'], list):
                num_issues = len(result['deliverables'])  # other checks have a list
            total_issues += num_issues
            issues_by_category[key] = num_issues

    # --- Prepare Chart Data for Chart.js --- #
    chart_labels = list(issues_by_category.keys())
    chart_data = list(issues_by_category.values())

    results['summary'] = {
        'total_deliverables': total_deliverables,
        'total_issues': total_issues,
        'issues_by_category': issues_by_category,
        'chart_labels': chart_labels,  # Add labels for the chart
        'chart_data': chart_data    # Add data for the chart
    }

    return results
