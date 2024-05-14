# Machine Learning Assignment 1

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

loan_new_data = pd.read_csv("loan_new.csv")
loan_old = pd.read_csv("loan_old.csv")


missing_values = loan_old.isnull().sum()
print("Missing Values:\n", missing_values)
print("There are missing values in Gender, Married, Dependents, Loan_Tenor, Credit_History, and Max_Loan_Amount. ")

numerical_features = loan_old.select_dtypes(include=['int64', 'float64'])
print("\nNumerical Features:\n", numerical_features.columns)

categorical_features = loan_old.select_dtypes(include=['object'])
print("\nCategorical Features:\n", categorical_features.columns)

numerical_features.agg(['mean', 'std'])
print("\nThe features have different scales.\nIncome and Coapplicant_Income have large values, which are in the thousands.")
print("Loan_Tenor has relatively smaller values, indicating months.\nCredit_History is a binary (0 or 1) feature.\nMax_Loan_Amount is in thousands.")

column_data_types = loan_old.dtypes
print(column_data_types)

# plotting the data
sns.set_palette("pastel", n_colors=9, color_codes=True)
sns.pairplot(loan_old)
plt.show()


loan_old.dropna(inplace=True)

loan_old.reset_index(drop=True, inplace=True)

missing_values2 = loan_old.isnull().sum()
print("Missing Values after deletion:\n", missing_values2)

X = loan_old.drop(['Max_Loan_Amount'], axis=1)
Y = loan_old[['Max_Loan_Amount']]

def encode(X):
    # Extracting categorical features
    XX = X.select_dtypes(include=["object"])
    XX.drop(["Loan_ID"], inplace=True, axis=1)

    categorical_features = XX.astype('str')

    encoded_categorical_features = pd.DataFrame()

    for column in categorical_features.columns:
        encoder = LabelEncoder()
        encoded_categorical_features[column] = encoder.fit_transform(X[column])
        X[column] = encoded_categorical_features[column]

    return X

def standard_scale(X):
    # Extracting numerical features
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns

    numerical_columns_to_standardize = []
    for col in numerical_columns:
        if col != "Credit_History":
            numerical_columns_to_standardize.append(col)

    # Standard scaling numerical features
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(X[numerical_columns_to_standardize])

    X[numerical_columns_to_standardize] = pd.DataFrame(numerical_data, columns=numerical_columns_to_standardize)
    return X



# encode and standardize the data
scaler = StandardScaler()
Y = scaler.fit_transform(Y)
X = encode(X)
X=standard_scale(X)


# linear regression model

# drop the loan ID
X1 = X.drop(["Loan_ID", "Loan_Status"], axis=1)

# split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=40)

reg = linear_model.LinearRegression()

# train the data
reg.fit(X_train, y_train)

# predict the test data
y_estimated = reg.predict(X_test)

mse = mean_squared_error(y_test, y_estimated)
print("mean squared error is: ", mse)

R2score = r2_score(y_test, y_estimated)
print("r2 score of the model: ", R2score)

# implementation of logistic regression
label_encoder = LabelEncoder()
Y_logistic = label_encoder.fit_transform(loan_old["Loan_Status"])
X_logistic = loan_old.drop(["Loan_Status"], axis=1)

# Encode the data
X_logistic = encode(X_logistic)
X_logistic = standard_scale(X_logistic)
X_logistic = X_logistic.drop(["Loan_ID", "Max_Loan_Amount"], axis=1)

# print(Y_logistic)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_logistic, Y_logistic, test_size=0.20,
                                                                    random_state=40)

def sigmoid(z):
     z = np.clip(z, -700, 700)
     return 1 / (1 + np.exp(-z))


def cost_function(X, theta, bias):
    z = np.dot(X, theta) + bias
    f_sigmoid = sigmoid(z)
    return f_sigmoid

def gradient_descent(iterations, alpha, X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    cost_history = []
    for iteration in range(iterations):
        z = np.dot(X, theta) + bias
        f_sigmoid = sigmoid(z)
        d_theta = (1 / m) * np.dot(X.T, (f_sigmoid - Y))
        d_bias = (1 / m) * np.sum(f_sigmoid - Y)
        theta -= alpha * d_theta
        bias -= alpha * d_bias
        cost = -(1 / m) * np.sum(Y * np.log(f_sigmoid) + (1 - Y) * np.log(1 - f_sigmoid))
        cost_history.append(cost)
        if iteration % math.ceil(iterations / 10) == 0:
            print(f"Iteration {iteration:4d}: Cost {cost_history[-1]}   ",
                  calculate_accuracy(y_test_log, predict(X_test_log, theta, bias)))
    return theta, bias, cost_history


def predict(X, theta, bias):
    predictions = cost_function(X, theta, bias)
    return (predictions >= 0.5).astype(int)


def calculate_accuracy(y_true, y_pred):
    # Ensure the length of y_true and y_pred is the same
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")

    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100

    return accuracy


# plotting the data
def plot_cost_history(cost_history):
    plt.plot(cost_history)
    plt.title('Cost (Error) by Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (Error)')
    plt.style.use('dark_background')
    plt.show()


# Train custom logistic regression
iterations = 100000
alpha = 0.01

theta, bias, cost_history = gradient_descent(iterations, alpha, X_train_log, y_train_log)
predicted_test_custom = predict(X_test_log, theta, bias)
# Accuracy on test set

accuracy_test_custom = calculate_accuracy(y_test_log, predicted_test_custom)
print("custom logistic accuracy test ", accuracy_test_custom, "%")
plot_cost_history(cost_history)

print("New Theta :", theta)
print("New Bias :", bias)


# preprocessing of the data
loan_new_data.describe()

missing_values_new = loan_new_data.isnull().sum()
print("Missing Values:\n", missing_values_new)
print("There are missing values in Gender, Dependents, Loan_Tenor, Credit_History. ")

numerical_features_new = loan_new_data.select_dtypes(include=['int64', 'float64'])
categorical_features_new = loan_new_data.select_dtypes(include=['object'])
print("\nNumerical Features:\n", numerical_features_new.columns)
print("\nCategorical Features:\n", categorical_features_new.columns)

numerical_features_new.agg(['mean', 'std'])
print("\nThe features have different scales.\nIncome and Coapplicant_Income have large values, which are in the thousands.")
print("Loan_Tenor has relatively smaller values, indicating months.\nCredit_History is a binary (0 or 1) feature.\n")

column_data_types = loan_new_data.dtypes
print(column_data_types)

loan_new_data.dropna(inplace=True)

loan_new_data.reset_index(drop=True, inplace=True)

loan_new = loan_new_data

loan_new = encode(loan_new)
loan_id = loan_new["Loan_ID"]
loan_new_final = loan_new.drop(["Loan_ID"], axis=1)

# linear regression prediction
y_estimated_linear = reg.predict(loan_new_final)
#print("Maximum Amount Estimiated For Every Apllicant: \n",y_estimated_linear)

y_estimated_logistic = predict(loan_new_final, theta, bias)
mapping = {1: 'Y', 0: 'N'}

# Use the replace method to map 1s and 0s to 'Y' and 'N'
vfunc = np.vectorize(mapping.get)
y_estimated_logistic = vfunc(y_estimated_logistic)

#print("Loan Status Predicted For Every Applicant: ",y_estimated_logistic)
loan_new_final.insert(loc=0, column='Loan_ID', value=loan_id)
loan_new_final['Maximum Amount Estimiated'] = y_estimated_linear
loan_new_final['Loan Status Predicted'] = y_estimated_logistic
print(loan_new_final.to_string())