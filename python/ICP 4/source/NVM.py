from sklearn import datasets    # Importing sklearn
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plot # Importing matplotlib
from sklearn import metrics
from sklearn.model_selection import train_test_split


iris = datasets.load_iris() # Importing data sets


model = GaussianNB()  # Fitting the nvm
model.fit(iris.data, iris.target)   # plotting the graph
expected = iris.target
predicted = model.predict(iris.data) # Fitting the model on the data


plot.plot(expected, predicted)     # plotting the expected and predicted values
print(expected)
print(predicted)
plot.show()
print(metrics.classification_report(expected, predicted)) # cross validation and comparing predicting and expected values
print(metrics.confusion_matrix(expected, predicted))   # Finding the confusion matrix
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
model.fit(X_train, Y_train)  # Training the model on the training data
Y_predicted = model.predict(X_test)  # Training the model on the test data
# Evaluating the Model based on Testing Part
print("Gaussian Model Accuracy is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)