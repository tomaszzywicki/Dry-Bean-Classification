from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def evaluate_accuracy(model, X, y, X_test, y_test):
    model.fit(X, y)
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return 