# Data Preprocessing

# 1. Importing the libraries
import numpy as np # Contains mathematical tools
import matplotlib.pyplot as plt # Contains plotting tools
import pandas as pd # Contains dataset import/manage tools

# 2. Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# 3. Divide the dataset into independent(X) and dependent(y) variables
X = dataset.iloc[:, :-1].values # Matrix of features/variables
y = dataset.iloc[:, 1].values # Matrix of outputs/prediction/result

# Data Manipulation and Cleaning
# 4. Take care of missing data
from sklearn.preprocessing import Imputer # Imputer is a class that takes care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])

# 5. Encode categorical data
# Encoding the Independent Variable
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""

# 6. Feature Scaling (Standardisation (StandardScaler) or Normalisation (Normalizer))
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)"""

# Test and Train Samples
# 7. Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split # cross_validation is deprecated and moved to model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Model Development - Simple Regression
# 8. Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 9. Predicting the Test set results
y_pred = regressor.predict(X_test)

# 10. Evaluate/Diagnose the Model - Mean Absolute Percentage Error (MAPE)

# 11. Model Visualization
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Green - Training set, Red - Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()