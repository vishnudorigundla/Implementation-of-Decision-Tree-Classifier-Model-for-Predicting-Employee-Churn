# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree Classifier and get the values of accuracy and data prediction.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: D.vishnu vardhan reddy
Register Number: 212221230023
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
#### 1.df.head()

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/c432e604-df63-43eb-a231-9cb039f44bc3)

#### 2. df.info()

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/8eef5726-b0c6-415a-b68d-241ca11f62f5)

#### 3. Null values

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/de27dcde-3868-473e-81fc-6cf4d94ad799)

#### 4. value_count() for left data

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/7e51a137-1b93-4f51-aa30-e12cf66e5b04)

#### 5. data.head() for salary

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/26794cce-2c73-49d1-b9df-12aedd4fb35e)

#### 6. x.head()

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/1c7d3baf-2bcd-453a-80a3-0222534d2b60)

#### 7. Accuracy value 

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/d9234479-b6da-4b39-8816-6c0273999603)

#### 8. prediction value

![image](https://github.com/vishnudorigundla/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94175324/a6fb4224-5b6f-4cb4-ae63-2982a641fe45)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
