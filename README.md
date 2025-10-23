Employee Attrition Classification

This project aims to predict employee attrition using machine learning techniques. By analyzing employee data, the model identifies factors contributing to employee turnover, enabling organizations to implement strategies to retain talent.

📁 Project Structure

Classifier.ipynb: Jupyter Notebook containing data preprocessing, model training, and evaluation.

best_model_employee_attition.joblib: Serialized machine learning model for predicting employee attrition.

requirements.txt: List of Python dependencies required to run the project.

streamlit_app.py: Streamlit application for interactive model deployment.

🚀 Getting Started
Prerequisites

Ensure you have Python 3.6+ installed. It's recommended to use a virtual environment.

Installation

Clone the repository:

git clone https://github.com/ShanukaAlahakoon/employee-attrition-classification.git
cd employee-attrition-classification

Install dependencies:

pip install -r requirements.txt

Running the Streamlit App

To launch the interactive web application:

streamlit run streamlit_app.py

This will start a local server and open the app in your default web browser.

📊 Model Overview

The project utilizes machine learning algorithms to classify employee attrition. Key steps include:

Data Preprocessing: Handling missing values, encoding categorical variables, and scaling features.

Model Training: Experimenting with various algorithms to identify the best-performing model.

Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.

🧪 Dependencies

The requirements.txt file includes necessary libraries such as:

pandas

numpy

scikit-learn

streamlit

matplotlib

seaborn
