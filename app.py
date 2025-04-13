# app.py

import os
import logging
import io
import tempfile
import openpyxl
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash # Added flash
from werkzeug.utils import secure_filename

# Import the analysis logic from analyzer.py
import analyzer

app = Flask(__name__)

# --- Configuration ---
# IMPORTANT: Set a secret key for session management and flash messages.
# Replace 'your_very_secret_key_here' with a real, strong secret key from env var or config.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_replace_me')

# Configure logging (can be done once here)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants moved or loaded in analyzer.py, keep UI related ones here
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 15 * 1024 * 1024 # 15 MB

# --- Helper Functions (Keep UI/Request related helpers) ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    session.pop('analysis_results', None) # Clear previous results
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_route(): # Renamed function to avoid conflict with module
    if 'file' not in request.files:
        logging.warning("Analyze request received without file part.")
        flash("No file part selected. Please choose a CSV file.", "error") # Flash message
        return redirect(url_for('index'))

    file = request.files['file']
    pod_filter = request.form.get('pod_filter', '').strip() or None
    bto_filter = request.form.getlist('bto_filter') or None

    if file.filename == '':
        logging.warning("Analyze request received with empty filename.")
        flash("No file selected. Please choose a CSV file.", "error") # Flash message
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        logging.warning(f"Invalid file type uploaded: {file.filename}")
        flash("Invalid file type. Only CSV files (.csv) are allowed.", "error") # Flash message
        return redirect(url_for('index'))

    temp_file_path = None # Initialize path variable
    try:
        # Check file size *before* saving to temp file
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > MAX_FILE_SIZE:
             logging.warning(f"File size exceeds limit: {file_length} bytes.")
             flash(f"File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024):.1f} MB.", "error") # Flash message
             return redirect(url_for('index'))
        file.seek(0)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix="_" + secure_filename(file.filename)) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        logging.info(f"File saved temporarily to: {temp_file_path}")

        # *** Call analysis logic from analyzer module ***
        analysis_results = analyzer.analyze_deliverables(temp_file_path, pod_filter, bto_filter)

        # Store results in session
        session['analysis_results'] = analysis_results

        # Check for errors/info returned by the analyzer
        if 'error' in analysis_results:
            logging.error(f"Analysis failed: {analysis_results['error']}")
            flash(f"Analysis Error: {analysis_results['error']}", "error")
            # Still render results page to show the error message contextually
            return render_template('results.html', analysis_results=None, error_message=analysis_results['error'])
        elif 'info' in analysis_results:
            logging.info(f"Analysis info: {analysis_results['info']}")
            flash(analysis_results['info'], "info") # Use info category
            return render_template('results.html', analysis_results=analysis_results, info_message=analysis_results['info'])
        # No specific warning message handling here, warnings logged in analyzer

        logging.info("Analysis successful, rendering results.")
        flash("Analysis completed successfully!", "success") # Success message
        return render_template('results.html', analysis_results=analysis_results)

    except Exception as e:
        logging.exception("Unexpected error during file processing or analysis.")
        flash(f"An unexpected server error occurred. Please try again later.", "error")
        # Redirect to index on unexpected errors during the process
        return redirect(url_for('index'))

    finally:
        # Ensure temporary file is removed
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"Removed temporary file: {temp_file_path}")
            except OSError as e:
                logging.error(f"Error removing temporary file {temp_file_path}: {e}")

@app.route('/export', methods=['GET'])
def export_results():
    analysis_results = session.get('analysis_results')

    # Check if results exist and are not an error/info message placeholder
    if not analysis_results or isinstance(analysis_results.get('error'), str) or isinstance(analysis_results.get('info'), str):
         logging.warning("Export requested but no valid analysis results found in session.")
         flash("No analysis results available to export. Please run an analysis first.", "warning")
         return redirect(url_for('index'))

    try:
        wb = openpyxl.Workbook()
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        logging.info("Generating Excel file...")

        for key, result_data in analysis_results.items():
            if not isinstance(result_data, dict) or 'deliverables' not in result_data:
                continue

            deliverables = result_data['deliverables']
            # Truncate sheet name safely and make it readable
            sheet_title_base = key.replace('_', ' ').title()
            sheet_name = (sheet_title_base[:28] + '...') if len(sheet_title_base) > 31 else sheet_title_base
            try:
                 ws = wb.create_sheet(title=sheet_name)
            except ValueError: # Handle potential duplicate truncated names
                 sheet_name = f"{sheet_name[:25]} ({key[:3]})"
                 ws = wb.create_sheet(title=sheet_name)


            headers = ["Deliverable Name"]
            is_missing_fields_check = key in ["missing_mandatory_fields", "missing_optional_fields"]
            # Check if deliverables is a dict (specific for missing fields)
            is_dict_structure = isinstance(deliverables, dict)

            if is_dict_structure:
                headers.append("Missing Fields")
            elif isinstance(deliverables, list) and deliverables and isinstance(deliverables[0], str) and ":" in deliverables[0]:
                 # Crude check if it looks like pre-formatted error strings from analyzer
                 headers = ["Issue Detail"] # Use a generic header

            ws.append(headers)

            if is_dict_structure:
                 if not deliverables:
                    ws.append(["No issues found in this category", ""])
                 else:
                    for deliverable_name, details in deliverables.items():
                        if isinstance(details, list): # Standard missing fields
                             ws.append([deliverable_name, ", ".join(details)])
                        else: # Handle potential error strings within the dict
                             ws.append([deliverable_name, str(details)])
            elif isinstance(deliverables, list):
                 if not deliverables:
                     ws.append(["No issues found in this category"])
                 else:
                    for item in deliverables:
                        ws.append([str(item)]) # Ensure it's a string
            else: # Handle unexpected format
                 ws.append([f"Unexpected data format for {key}"])

            # Auto-adjust column widths
            for col_idx, column_cells in enumerate(ws.columns, 1):
                max_length = 0
                column_letter = openpyxl.utils.get_column_letter(col_idx)
                for cell in column_cells:
                    try:
                        if cell.value:
                            cell_length = len(str(cell.value))
                            if cell_length > max_length:
                                max_length = cell_length
                    except:
                        pass
                adjusted_width = min((max_length + 2) * 1.1, 60) # Adjust factor, add max width cap
                ws.column_dimensions[column_letter].width = adjusted_width

        excel_stream = io.BytesIO()
        wb.save(excel_stream)
        excel_stream.seek(0)

        logging.info("Excel file generated successfully.")
        return send_file(
            excel_stream,
            as_attachment=True,
            download_name='analysis_results.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logging.exception("Error generating Excel file.")
        flash("An server error occurred while generating the Excel report.", "error")
        return redirect(url_for('index'))

# --- Main Execution ---
if __name__ == "__main__":
    # For development only - use Gunicorn/Waitress for production
     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False) # Keep debug=False usually
