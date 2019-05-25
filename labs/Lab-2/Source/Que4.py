import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

# Load the train DataFrames using pandas
train = pd.read_csv('C:/Users/laksh/PycharmProjects/Python_Lesson5/data/train.csv')

print("Train_Set")
print(train.head())
print("\n")

#print("Train_Set description")
#print(train.describe())
#print("\n")

print(train.columns.values)

# For identifying no.of null values in the train set
train.isna().head()

print("NULL values in the train ")
print(train.isna().sum())
print("\n")

# Fill missing or null values with mean column values in the train set
train.fillna(train.mean(), inplace=True)

print(train.isna().sum())

train.info()
#removing the features not correlated to the target class,
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])


train.info()
print(train.head())


feature_cols = ['PassengerId' ,'Pclass', 'Age' ,'SibSp' ,'Parch' ,'Fare']
from sklearn.svm import SVC

X = train[feature_cols]
y = train.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("Variance score: %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

