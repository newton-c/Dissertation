from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import random

random.seed(3476)
# importing the dataset
fp = '~/Desktop/dissertation/predict/data/newton_data4-30-2020.csv'
data = pd.read_csv(fp, encoding='latin_1')
print(data.columns)
data = data[data['hom_rate'].notna()]

fp = '~/Desktop/dissertation/predict/data/F&Lethnic.csv'
ef = pd.read_csv(fp)
cols = ['year', 'mtnest', 'ef', 'ccode']
ef = ef[cols]
data = pd.merge(data, ef,
                how="left", on=["ccode", "year"])

# Null model
y = data.hom_rate # target feature

#features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'rgdppc', 
#      'v2x_libdem', 'drugs_any', 'gems_any', 'agriculture_any', 
#      'minerals_any', 'fuel_any']

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem']

X = data[features] # imput features


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# imputing NaNs
my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

X_train_imputed.columns = X_train.columns
X_test_imputed.columns = X_test.columns

# feature scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imputed)
X_test_sc = sc.transform(X_test_imputed)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_sc, y_train)
preds = lin_reg.predict(X_test_sc)

print('\nNull Models\n')
print('Linear Regression\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', lin_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# stochastic gradient descent
sgd_reg = SGDRegressor(max_iter=500, penalty=None, eta0=0.001)
sgd_reg.fit(X_train_sc, y_train)
preds = sgd_reg.predict(X_test_sc)

print('\nStocastic Gradient Descent\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', sgd_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# random forest
rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
rf_reg.fit(X_train_sc, y_train)
preds = rf_reg.predict(X_test_sc)

print('\nRandom Forest\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

#print('Coefficients: \n', rf_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# Social Disorganization Model

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'ef']

X = data[features] # imput features

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# imputing NaNs
my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

X_train_imputed.columns = X_train.columns
X_test_imputed.columns = X_test.columns

# feature scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imputed)
X_test_sc = sc.transform(X_test_imputed)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_sc, y_train)
preds = lin_reg.predict(X_test_sc)

print('\nSocial Disorganization Models\n')
print('Linear Regression\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', lin_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# stochastic gradient descent
sgd_reg = SGDRegressor(max_iter=500, penalty=None, eta0=0.001)
sgd_reg.fit(X_train_sc, y_train)
preds = sgd_reg.predict(X_test_sc)

print('\nStocastic Gradient Descent\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', sgd_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# random forest
rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
rf_reg.fit(X_train_sc, y_train)
preds = rf_reg.predict(X_test_sc)

print('\nRandom Forest\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

#print('Coefficients: \n', rf_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# Political Economy Model
features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'rgdppc']

X = data[features] # imput features

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# imputing NaNs
my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

X_train_imputed.columns = X_train.columns
X_test_imputed.columns = X_test.columns

# feature scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imputed)
X_test_sc = sc.transform(X_test_imputed)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_sc, y_train)
preds = lin_reg.predict(X_test_sc)

print('\nPolitical Economy Models\n')
print('Linear Regression\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', lin_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# stochastic gradient descent
sgd_reg = SGDRegressor(max_iter=500, penalty=None, eta0=0.001)
sgd_reg.fit(X_train_sc, y_train)
preds = sgd_reg.predict(X_test_sc)

print('\nStocastic Gradient Descent\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', sgd_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# random forest
rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
rf_reg.fit(X_train_sc, y_train)
preds = rf_reg.predict(X_test_sc)

print('\nRandom Forest\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

#print('Coefficients: \n', rf_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# Organizational model

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'drugs_any', 'gems_any', 'agriculture_any', 
      'minerals_any', 'fuel_any']

X = data[features] # imput features

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# imputing NaNs
my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

X_train_imputed.columns = X_train.columns
X_test_imputed.columns = X_test.columns

# feature scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imputed)
X_test_sc = sc.transform(X_test_imputed)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_sc, y_train)
preds = lin_reg.predict(X_test_sc)

print('\nOrganizational Models\n')
print('Linear Regression\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', lin_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# stochastic gradient descent
sgd_reg = SGDRegressor(max_iter=500, penalty=None, eta0=0.001)
sgd_reg.fit(X_train_sc, y_train)
preds = sgd_reg.predict(X_test_sc)

print('\nStocastic Gradient Descent\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

print('Coefficients: \n', sgd_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))

# random forest
rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
rf_reg.fit(X_train_sc, y_train)
preds = rf_reg.predict(X_test_sc)

print('\nRandom Forest\n')
print('Mean absoulte error: %.2f'
        % mean_absolute_error(y_test, preds))

#print('Coefficients: \n', rf_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, preds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, preds))
      