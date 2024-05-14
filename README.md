# loan-eligibility-prediction-python

Overview:

This Python program is designed to automate the process of validating customers' eligibility for interest-free home loans offered by a housing finance company. It utilizes machine learning models, specifically linear regression and logistic regression, to predict loan decisions and amounts based on customers' details provided in the loan application form.

<br>

Program Features:

Data Analysis: The program performs analysis on the provided dataset to identify missing values, determine feature types (categorical or numerical), check for scale consistency among numerical features, and visualize relationships between numerical columns.

Data Preprocessing: It preprocesses the data by removing records with missing values, separating features and targets, shuffling and splitting the data into training and testing sets, encoding categorical features and targets, and standardizing numerical features.

Linear Regression Model: The program fits a linear regression model to predict the loan amount using scikit-learn's LinearRegression.

Evaluation: It evaluates the linear regression model's performance using scikit-learn's R^2 score.

Logistic Regression Model: The program implements logistic regression from scratch using gradient descent to predict the loan status.

Accuracy Calculation: It includes a function to calculate the accuracy of the logistic regression model from scratch.

Prediction on New Data: The program preprocesses and predicts loan amounts and status for new applicant data using the trained models.

<br>

Datasets:

loan_old.csv: Contains 614 records of applicants' data with 10 feature columns and 2 target columns (maximum loan amount and loan acceptance status).
loan_new.csv: Contains 367 records of new applicants' data with the same 10 feature columns.


team member: @https://github.com/malakg1
