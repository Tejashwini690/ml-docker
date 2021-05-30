import pandas
import joblib
data = pandas.read_csv('SD.csv')
X = data['YearsExperience']
Y = data['Salary']
X = X.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model =  LinearRegression()
model.fit(X,Y) 
joblib.dump(model,'new_model.pkl')
train_model = joblib.load("new_model.pkl")
experience = input("enter your experience to predict your salary:")
salary = train_model.predict([[float(experience)]])
print("your expected salary is : {}".format(salary))