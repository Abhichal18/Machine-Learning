import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train=pd.read_csv('train.csv')
print('---- Train Data Imported Succesfully ----')
print(train.head())

X=train.iloc[:,:-1].values
y=train.iloc[:,1].values

# splitting the data into test and train so that we can find the mean squared error
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

linear_regression=LinearRegression()
linear_regression.fit(X_train,y_train)

print("---- Model is Trained ----")

regression_line= linear_regression.coef_*X+linear_regression.intercept_

y_predicted=linear_regression.predict(X_test)

mse=mean_squared_error(y_test,y_predicted)
print('\nMean Squared Error is ',mse)

test=pd.read_csv('testX.csv')
print('\n---- Test Data Imported Successfully ----')
print(test.head())
X_test=test.values
y_predicted=linear_regression.predict(X_test)
print('\n---- Prediction Successful for Test Data ----')


X_test=test['Xts']
submission = pd.DataFrame({'X_test': X_test, 'Y_test': y_predicted}) 
print(submission.head())
submission.to_csv('submission.csv',index=False)