import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car data.csv')
# Separate numeric and non-numeric columns
numeric_cols = car_dataset.select_dtypes(include=['number']).columns
non_numeric_cols = car_dataset.select_dtypes(exclude=['number']).columns
# Handling missing values in numeric columns with mean imputation
imputer_numeric = SimpleImputer(strategy='mean')
car_dataset[numeric_cols] = imputer_numeric.fit_transform(car_dataset[numeric_cols])
# Handling missing values in non-numeric columns with the most frequent value imputation
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
car_dataset[non_numeric_cols] = imputer_non_numeric.fit_transform(car_dataset[non_numeric_cols])
fig=plt.hist(car_dataset.Car_Name)
fig=plt.scatter(car_dataset.Car_Name,car_dataset.Selling_Price) 
plt.xlabel('Car_Name')
plt.ylabel('Selling_Price')
# inspecting the first 5 rows of the dataframe
car_dataset.head()
# checking the number of rows and columns
car_dataset.shape
# getting some information about the dataset
car_dataset.info()
# checking the number of missing values
car_dataset.isnull().sum()
# checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())
# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
car_dataset.head()
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)
# loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)
# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
# prediction on Training data
test_data_prediction = lin_reg_model.predict(X_test)
# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
# loading the linear regression model
lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)
# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
# prediction on Training data
test_data_prediction = lass_reg_model.predict(X_test)
# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
#With R2
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R2 Error : ", error_score)
MAE = mean_absolute_error(Y_test, test_data_prediction)
RMSE = np.sqrt(MAE)

print("Mean Absolute Error (MAE):", MAE)
print("Root Mean Squared Error (RMSE):", RMSE)
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()