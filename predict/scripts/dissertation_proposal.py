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

# For Neural Net
import tensorflow as tf
from tensorflow import keras

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

# linear regression
def lin_models(X_train, y_train, X_test, y_test, model_name):
      pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
          ('std_scaler', StandardScaler()),
          ('lin_reg', LinearRegression())])
      pipe.fit(X_train, y_train)
      preds = pipe.predict(X_test)
      print('\nLinear Regression: %s\n' % model_name)
      print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
      print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
      print('Coefficient of determination: %.2f' % r2_score(y_test, preds))

lin_models(X_train, y_train, X_test, y_test, "Null Model OLS")

# stochastic gradient descent
def sgd_models(X_train, y_train, X_test, y_test, max_iter, 
        penalty, eta0, model_name):
        """
        """
        pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
           ('std_scaler', StandardScaler()),
           ('sgd_reg', SGDRegressor(max_iter=max_iter, 
               penalty=penalty, eta0=eta0))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        print('\nStocastic Gradient Descent %s\n' % model_name) 
        print('Mean absoulte error: %.2f' % mean_absolute_error(y_test, preds))
        print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
        print('Coefficient of determination: %.2f' % r2_score(y_test, preds))

sgd_models(X_train, y_train, X_test, y_test, 
        500, None, 0.001, "Null Model SGD")

# random forest
def rf_models(X_train, y_train, X_test, y_test, n_estimators,
      random_state, model_name):
      """
      """
      pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=n_estimators,
            random_state=random_state))])
      pipe.fit(X_train, y_train)
      preds = pipe.predict(X_test)
      print('\nRandom Forest: %s\n' % model_name)
      print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
      print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
      print('Coefficient of determination: %.2f' % r2_score(y_test, preds))


rf_models(X_train, y_train, X_test, y_test, 200, 0, "Null Model")

# Social Disorganization Model

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'ef']

X = data[features] # imput features

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_models(X_train, y_train, X_test, y_test, 
      200, 0, "Social Disorganization Model")

# Political Economy Model
features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'rgdppc']

X = data[features] # imput features

# Organizational model

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem', 'drugs_any', 'gems_any', 'agriculture_any', 
      'minerals_any', 'fuel_any']

X = data[features] # imput features

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_models(X_train, y_train, X_test, y_test, 
      200, 0, "Organizational Model")

# Trying a simple neural net (Regression MLP)

y = data.hom_rate 

features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
      'v2x_libdem']

X = data[features] 


# splitting the data
X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, random_state=42)

# so we have training, testing, and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_valid_imputed = pd.DataFrame(my_imputer.fit_transform(X_valid))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

X_train_imputed.columns = X_train.columns
X_valid_imputed.columns = X_valid.columns
X_test_imputed.columns = X_test.columns

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train_imputed)
X_valid = sc.transform(X_valid_imputed)
X_test = sc.transform(X_test_imputed)

# building the net
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.01))
history = model.fit(X_train, y_train, epochs=20,
        validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(mse_test)

preds = model.predict(X_test)
print('\nNeural Net:\n')
print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
print('Coefficient of determination: %.2f' % r2_score(y_test, preds))

