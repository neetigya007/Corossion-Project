from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Random Forest models
rf_models = {
    "blank": joblib.load("rf_model_blank.pkl"),
    "ppm": joblib.load("rf_model_ppm.pkl"),
    "icorr": joblib.load("rf_model_icorr.pkl"),
    "icorrnew": joblib.load("rf_model_icorrnew.pkl")
}

# Load Decision Tree models
dt_models = {
    "blank": joblib.load("decision_tree_ecorr_blank.pkl"),
    "ppm": joblib.load("decision_tree_ecorr_500ppm.pkl"),
    "icorr": joblib.load("decision_tree_icorr.pkl"),
    "icorrnew": joblib.load("decision_tree_icorrnew.pkl")
}

# Function to calculate inhibitor efficiency
def calculate_efficiency(icorr_blank, icorr_ppm):
    return ((icorr_blank - icorr_ppm) / icorr_blank) * 100 if icorr_blank != 0 else 0

@app.route('/')
def home():
    return render_template('index_2.html')  # Ensure index.html has a model type selector

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    temp = float(request.form['temp'])
    flowrate = int(request.form['flowrate'])
    model_type = request.form.get('model_type', 'rf')  # 'rf' or 'dt'

    # Select model set
    models = rf_models if model_type == 'rf' else dt_models

    # Input preparation
    X_input = np.array([[temp, flowrate]])

    # Model predictions
    pred_blank = models["blank"].predict(X_input)[0]
    pred_ppm = models["ppm"].predict(X_input)[0]
    pred_icorr = models["icorr"].predict(X_input)[0]
    pred_icorrnew = models["icorrnew"].predict(X_input)[0]

    # Calculate efficiency
    inhibitor_efficiency = calculate_efficiency(pred_icorr, pred_icorrnew)

    return jsonify({
        "Model Type": "Random Forest" if model_type == 'rf' else "Decision Tree",
        "Predicted Ecorr (Blank)": round(pred_blank, 2),
        "Predicted Ecorr (500ppm)": round(pred_ppm, 2),
        "Predicted Icorr (Blank)": round(pred_icorr, 5),
        "Predicted Icorr (500ppm)": round(pred_icorrnew, 5),
        "Inhibitor Efficiency (%)": round(inhibitor_efficiency, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
