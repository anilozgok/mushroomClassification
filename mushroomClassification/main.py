import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# data load and check
dataset = pd.read_csv("mushrooms.csv")
print(dataset.head())

# understanding dataset
clasess = dataset["class"].value_counts()
print(clasess)

# plot bar for visualizing
plt.bar('Edible', clasess['e'])
plt.bar('Poisonous', clasess['p'])
plt.show()

# creating variables for futures
X = dataset.loc[:, ["cap-shape", "cap-color", "ring-number", "ring-type"]]
y = dataset.loc[:, "class"]

# label encoding
labelEncoder = LabelEncoder()
for item in X.columns:
    X[item] = labelEncoder.fit_transform(X[item])
y = labelEncoder.fit_transform(y)

# splitting train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# creating model objects
logisticRegression_Model = LogisticRegression()
ridgeClass_Model = RidgeClassifier()
decisionTree_Model = DecisionTreeClassifier()
gaussianNB_Model = GaussianNB()
MLP_Model = MLPClassifier()

# training models
logisticRegression_Model.fit(X_train, y_train)
ridgeClass_Model.fit(X_train, y_train)
decisionTree_Model.fit(X_train, y_train)
gaussianNB_Model.fit(X_train, y_train)
MLP_Model.fit(X_train, y_train)

# making predictions
prediction_logReg = logisticRegression_Model.predict(X_test)
prediction_ridClass = ridgeClass_Model.predict(X_test)
prediction_decTree = decisionTree_Model.predict(X_test)
prediction_GNB = gaussianNB_Model.predict(X_test)
prediction_MLP = MLP_Model.predict(X_test)

# comparing performances
print(classification_report(y_test, prediction_logReg))
print(classification_report(y_test, prediction_ridClass))
print(classification_report(y_test, prediction_decTree))
print(classification_report(y_test, prediction_GNB))
print(classification_report(y_test, prediction_MLP))

# evaluating
randomForest_Model = RandomForestClassifier()
randomForest_Model.fit(X_train, y_train)
prediction_ranFor = randomForest_Model.predict(X_test)
print(classification_report(y_test, prediction_ranFor))
