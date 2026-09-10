import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained logistic regression model
with open('logistic.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML template with embedded CSS styling and categorical form controls
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Risk Predictor</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            width: 100%;
            max-width: 600px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 24px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        input, select {
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--primary);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .safe { background-color: #d1fae5; color: #065f46; }
        .at-risk { background-color: #fef3c7; color: #92400e; }
        .high-risk { background-color: #fee2e2; color: #991b1b; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Prediction</h2>
        <form method="POST" class="form-grid">
            <div class="form-group">
                <label for="attendance">Attendance (%)</label>
                <input type="number" step="any" name="attendance" required placeholder="e.g. 85">
            </div>
            <div class="form-group">
                <label for="study_hours">Study Hours (Weekly)</label>
                <input type="number" step="any" name="study_hours" required placeholder="e.g. 10">
            </div>
            <div class="form-group">
                <label for="past_failures">Past Failures</label>
                <input type="number" name="past_failures" required placeholder="e.g. 0">
            </div>
            <div class="form-group">
                <label for="assignments_completed_pct">Assignments Completed (%)</label>
                <input type="number" step="any" name="assignments_completed_pct" required placeholder="e.g. 90">
            </div>
            <div class="form-group">
                <label for="parental_education">Parental Education</label>
                <select name="parental_education" required>
                    <option value="0">High School</option>
                    <option value="1">Bachelor's Degree</option>
                    <option value="2">Master's / Higher</option>
                </select>
            </div>
            <div class="form-group">
                <label for="family_income">Family Income Level</label>
                <select name="family_income" required>
                    <option value="0">Low</option>
                    <option value="1">Medium</option>
                    <option value="2">High</option>
                </select>
            </div>
            <div class="form-group">
                <label for="extracurricular">Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label for="internet_access">Internet Access</label>
                <select name="internet_access" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label for="previous_grade">Previous Grade</label>
                <input type="number" step="any" name="previous_grade" required placeholder="e.g. 78">
            </div>
            <div class="form-group">
                <label for="final_score">Final Score</label>
                <input type="number" step="any" name="final_score" required placeholder="e.g. 82">
            </div>
            <button type="submit">Predict Risk Status</button>
        </form>

        {% if prediction %}
            <div class="result-card {{ css_class }}">
                Predicted Status: {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    css_class = ""
    
    if request.method == 'POST':
        try:
            # Extract features in the exact order trained by the model
            features = [
                float(request.form['attendance']),
                float(request.form['study_hours']),
                float(request.form['past_failures']),
                float(request.form['assignments_completed_pct']),
                float(request.form['parental_education']),
                float(request.form['family_income']),
                float(request.form['extracurricular']),
                float(request.form['internet_access']),
                float(request.form['previous_grade']),
                float(request.form['final_score'])
            ]
            
            # Reshape for single prediction
            features_array = np.array([features])
            pred = model.predict(features_array)[0]
            prediction = str(pred)
            
            # Map styling classes based on categories ('Safe', 'At-Risk', 'High-Risk')
            if prediction == 'Safe':
                css_class = 'safe'
            elif prediction == 'At-Risk':
                css_class = 'at-risk'
            else:
                css_class = 'high-risk'
                
        except Exception as e:
            prediction = f"Error in processing input: {str(e)}"
            css_class = 'high-risk'

    return render_template_string(HTML_TEMPLATE, prediction=prediction, css_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)
