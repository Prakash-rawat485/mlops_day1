import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import joblib

df=pd.read_csv(r"D:\MY Projects (github)\mlops_day1\data\data.csv")
X,y=df[["TV","Radio","Newspaper"]],df[["Sales"]]
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)



model=LinearRegression()
model.fit(Xtrain,ytrain)

ypred=model.predict(Xtest)
r2=r2_score(ytest, ypred)
rmse=np.sqrt(mean_squared_error(ytest, ypred))

print("R²:", r2)
print("RMSE:", rmse)

# Take my trained model and save it as model.pkl.

joblib.dump(model, r"D:\MY Projects (github)\mlops_day1\models\linear_regression_model.pkl")
