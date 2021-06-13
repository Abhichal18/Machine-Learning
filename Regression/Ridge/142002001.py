import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train=pd.read_csv('train.csv')
print('---- Train Data Imported Succesfully ----')
print(train.head())


X=train.iloc[:,:-1].values
y=train.iloc[:,1].values

# splitting the data into test and train so that we can find the mean squared error
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print('\nWhich cross validation technique you wanna use? Choose 1 or 2')
print('1. Leave One Out')
print('2. K fold')
user_input=int(input())

if user_input==1:
	model=RidgeCV(alphas=[1e-3,1e-2,1e-1,1],cv=None,fit_intercept=True).fit(X_train,y_train)
	print("\n---- Model is Trained ----")
	y_predicted=model.predict(X_test)
	mse=mean_squared_error(y_test,y_predicted)
	print('\nMean Squared Error is ',mse)

elif user_input==2:
	model=RidgeCV(alphas=[1e-3,1e-2,1e-1,1],cv=5,fit_intercept=True).fit(X_train,y_train)
	print("\n---- Model is Trained ----")
	y_predicted=model.predict(X_test)
	mse=mean_squared_error(y_test,y_predicted)
	print('\nMean Squared Error is ',mse)

else:
	print('\nYou Entered wrong Input. Try running the program again.')


test=pd.read_csv('testX.csv')
print('\n---- Test Data Imported Successfully ----')

print(test.head())
X_test=test.values
y_predicted=model.predict(X_test)
print('\n---- Prediction Successful for Test Data ----')

X_test=test['Xts']
submission = pd.DataFrame({'X_test': X_test, 'Y_test': y_predicted}) 
print(submission.head())
print('\n---- A submission.csv file is generated ----')
submission.to_csv('submission.csv',index=False)