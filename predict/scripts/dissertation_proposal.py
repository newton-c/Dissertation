from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem']

X = data[features] # imput features


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Pipeline
#model_pipeline = Pipeline([
#    ('imputer', SimpleImputer(strategy="median")),
#    ('std_scaler', StandardScaler()),
#])

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
def lin_models(X_train, y_train, X_test, y_test, model_name):
      lin_reg = LinearRegression() 
      lin_reg.fit(X_train, y_train)
      preds = lin_reg.predict(X_test)
      print('\nLinear Regression: %s\n' % model_name)
      print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
      print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
      print('Coefficient of determination: %.2f' % r2_score(y_test, preds))

lin_models(X_train_sc, y_train, X_test_sc, y_test, "Null Model OLS")

# stochastic gradient descent
def sgd_models(X_train, y_train, X_test, y_test, max_iter, 
        penalty, eta0, model_name):
        sgd_reg = SGDRegressor(max_iter=max_iter, penalty=penalty, eta0=eta0)
        sgd_reg.fit(X_train, y_train)
        preds = sgd_reg.predict(X_test)
        print('\nStocastic Gradient Descent\n')
        print('Mean absoulte error: %.2f' % mean_absolute_error(y_test, preds))
        print('Coefficients: \n', sgd_reg.coef_)
        print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
        print('Coefficient of determination: %.2f' % r2_score(y_test, preds))

sgd_models(X_train_sc, y_train, X_test_sc, y_test, 
        500, None, 0.001, "Null Model SGD")

# random forest
def rf_models(X_train, y_train, X_test, y_test, n_estimators,
      random_state, model_name):
      rf_reg = RandomForestRegressor(n_estimators=n_estimators,
            random_state=random_state)
      rf_reg.fit(X_train, y_train)
      preds = rf_reg.predict(X_test)
      print('\nRandom Forest: %s\n' % model_name)
      print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
      print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
      print('Coefficient of determination: %.2f' % r2_score(y_test, preds))


rf_models(X_train_sc, y_train, X_test_sc, y_test, 200, 0, "Null Model")

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

rf_models(X_train_sc, y_train, X_test_sc, y_test, 
      200, 0, "Social Disorganization Model")

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

rf_models(X_train_sc, y_train, X_test_sc, y_test, 
      200, 0, "Political Economy Model")

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

rf_models(X_train_sc, y_train, X_test_sc, y_test, 
      200, 0, "Organizational Model")
