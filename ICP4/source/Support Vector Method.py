import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets   # Importing packages
from sklearn import metrics
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() # Importing the datasets


from sklearn.svm import SVC

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

svm = SVC()  # Fitting the model
expected = iris.target

# Training the Model on Testing Set

svm.fit(X_train, y_train)
print(expected)
# Making Prediction based on X, Y
predicted_label = svm.predict(iris.data)
print(predicted_label)


print(metrics.confusion_matrix(expected, predicted_label))   # confusion matrix


print(metrics.classification_report(expected, predicted_label)) # Cross Validation compare the predicted and expected values

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train))) # Evaluating the model on the training model
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test))) # Evaluating model on test part