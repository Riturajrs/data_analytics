import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def predict_state(classifier, value):
    prediction = classifier.predict(np.array([[value]]))
    return "active" if prediction[0] == 1 else "inactive"


def get_model(states, values):
    state_to_num = {"active": 1, "inactive": 0}
    y = np.array([state_to_num[state] for state in states])
    X = np.array(values).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return classifier
