Real-Time Fraud Detection System 🛡️
This project provides an end-to-end solution for detecting fraudulent credit card transactions in real-time. It leverages a powerful XGBoost model trained on a large-scale transactional dataset and exposes the prediction functionality through a REST API and an interactive web interface.

The core challenge with fraud detection is the severe class imbalance in the data—fraudulent transactions are rare. This project addresses this by using the SMOTE technique and evaluates the model's performance using metrics suitable for imbalanced data, such as the Precision-Recall curve. To ensure transparency and trust, model predictions are explained using SHAP (SHapley Additive exPlanations).

🏗️ System Architecture
The application follows a simple, robust architecture for real-time inference:

Plaintext

User Input -> Web UI / API -> Data Preprocessing -> XGBoost Model -> Prediction (Fraud/Not Fraud) -> SHAP Explanation -> Display Result
✨ Key Features
High-Performance Model: Employs an XGBoost classifier, known for its speed and accuracy.

Imbalance Handling: Uses SMOTE (Synthetic Minority Over-sampling Technique) to create a balanced training set, leading to a more robust model.

Explainable AI (XAI): Integrates SHAP to provide clear, human-understandable explanations for every prediction.

Dual Interface: Accessible via both a REST API for programmatic access and a user-friendly Web UI for interactive demonstrations.

🛠️ Tech Stack
Backend: Python 3.9

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, imblearn

Model Interpretability: SHAP

API Server: FastAPI

Web Interface: Streamlit

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.8 or higher

Git command-line tools

2. Clone the Repository
Open your terminal and clone the project repository.

Bash

git clone https://github.com/Vignesh-pes/real-time-fraud-detection.git
cd real-time-fraud-detection
3. Create and Activate a Virtual Environment
Using a virtual environment is highly recommended to avoid conflicts with other projects.

Bash

# Create the virtual environment folder
python -m venv venv

# Activate the environment on Windows
.\venv\Scripts\activate

# On macOS/Linux, use: source venv/bin/activate
4. Install Dependencies
Install all the required Python packages using the requirements.txt file.

Bash

pip install -r requirements.txt
5. Download and Place the Dataset
The dataset used for training is not included in the repository due to its size.

Download Link: IEEE-CIS Fraud Detection Dataset

After downloading, create a directory named data in the project's root folder and place the train_transaction.csv and test_transaction.csv files inside it.

🏃‍♀️ How to Run the Application
You can run the API server and the web interface separately.

1. Run the API Server
The API provides the core prediction service. To start it, run the following command from the project's root directory:

Bash

# This command starts the FastAPI server
uvicorn app:app --reload
After running the command, your terminal will display logs indicating that the Uvicorn server has started successfully. You should see lines confirming that the application startup is complete and the server is listening for requests on http://127.0.0.1:8000.

2. Run the Web UI
The Streamlit interface provides an easy way to interact with the model. To launch it, open a new terminal and run:

Bash

streamlit run ui.py
This will automatically open a new tab in your web browser. The page will be titled "Real-Time Fraud Detection System" and will feature input fields for transaction details (like transaction amount, product code, card type, etc.) and a "Predict" button to submit the data for analysis.

📊 Model and Results
Model Performance
The XGBoost model's performance was evaluated using the Precision-Recall (PR) curve, which is ideal for imbalanced classification tasks. A successful model will have a PR curve that stays high and close to the top-right corner of the plot, indicating that it can achieve high precision while also identifying a high percentage of actual fraudulent cases (high recall). The Area Under the PR Curve (AUC-PR) provides a single score to summarize this performance.

Model Interpretation with SHAP
SHAP was used to understand which features have the most impact on predictions.

Global Feature Importance: A SHAP summary plot is used to rank features by their overall impact on the model's output. For this project, features like TransactionAmt, ProductCD, and various card-related attributes (card1, card4, etc.) are typically the most influential. The plot shows a distribution of SHAP values for each feature, revealing not just their importance but also whether high or low values of a feature tend to indicate fraud.

Individual Predictions: For any single prediction, a SHAP force plot provides a detailed explanation. It visualizes features as forces that either "push" the prediction towards a "Fraud" classification (shown in red) or towards "Not Fraud" (shown in blue). The size of the bar for each feature indicates the magnitude of its contribution to the final decision.

📝 API Usage
The API has the following endpoint for making predictions.

POST /predict
Send a POST request with a JSON object containing the transaction details.

URL: http://127.0.0.1:8000/predict

Method: POST

Body: A JSON object with transaction features.

Example Request:
JSON

{
  "TransactionDT": 86400,
  "TransactionAmt": 68.5,
  "ProductCD": "W",
  "card4": "discover",
  "card6": "credit",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com"
}
Example Success Response:
JSON

{
  "prediction": "Fraud",
  "probability": 0.89,
  "explanation": {
    "feature_1": 0.25,
    "feature_2": -0.15
  }
}
📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.