import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Data shape", X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state = 42, stratify= y)

bag_model = BaggingClassifier(estimator = DecisionTreeClassifier(), n_estimators=50, random_state = 42)
bag_model.fit(X_train,y_train)
y_pred_bag = bag_model.predict(X_test)
print("\n=== Bagging Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("classification report:\n", classification_report(y_test, y_pred_bag))

boost_model = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1), n_estimators=50, random_state= 42)
boost_model.fit(X_train,y_train)
y_pred_boost = boost_model.predict(X_test)

print("\n=== Boosting Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_boost))
print("classification report:\n", classification_report(y_test, y_pred_boost))

print("\n confusion matrix - Bagging:\n", confusion_matrix(y_test, y_pred_bag))
print("\n confusion matrix - Boosting:\n", confusion_matrix(y_test, y_pred_boost))