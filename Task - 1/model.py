import pickle
import pandas as pd


data=pd.read_csv('student.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X,y)


pickle.dump(regress,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[10]]))
