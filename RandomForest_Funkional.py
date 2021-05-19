import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy import stats


def random_forest(data, trees, max_depth):
    """Random forest process."""
    data_premuted = np.random.permutation(data)  # falls Daten nach y sortiert abgespeichert sind
    X_train, y_train, X_test, y_test = split_data(data_premuted)
    fittet_trees = fit(X_train, y_train, trees, max_depth)
    y_pred_train_list, y_pred_test_list = predict(fittet_trees, X_train, y_train, X_test, y_test)
    majority_vote(y_pred_train_list, y_pred_test_list, y_train, y_test)


def split_data(data, test_size=0.33):
    """Splits data in train and test data."""  # Die Testdaten sollen nicht in der Schleife permutiert werden
    z = int(len(data) * test_size)
    X_train = data[z:, 0:2]
    y_train = data[z:, 2]
    X_test = data[:z, 0:2]
    y_test = data[:z, 2]
    return X_train, y_train, X_test, y_test


def permut(X_train, y_train):
    """Permutiert die Trainingsdaten."""
    z = int(len(X_train))
    perm = np.random.permutation(z)
    X_train_permuted = X_train[perm, :]
    y_train_permuted = y_train[perm, ]
    return X_train_permuted, y_train_permuted


def fit(X_train, y_train, trees, max_depth):
    """Trainiert die Trainingsdaten mit der Anzahl "trees" und der Tiefe "max_depth"."""
    tree = DecisionTreeClassifier(max_depth=max_depth)
    fittet_trees = []
    for t in range(trees):
        # X_train_perm, X_test_perm, y_train_perm, y_test_perm = permut(data)
        X_train_permuted, y_train_permuted = permut(X_train, y_train)
        z = int(len(X_train_permuted) / 2)
        X_sub_train = X_train_permuted[:z, [0, 1]]
        y_sub_train = y_train_permuted[:z]
        fit_ = tree.fit(X_sub_train, y_sub_train)
        fittet_trees.append(fit_)
    return fittet_trees


def predict(fittet_trees, X_train, y_train, X_test, y_test):
    """Macht die Vorhersage und gibt y predicted für Trainings- und Testdaten wieder."""
    y_pred_train_list, y_pred_test_list = [], []
    for t in fittet_trees:
        # X_train_perm, y_train_perm = permut(X_train, y_train)
        y_pred_train = t.predict(X_train)
        y_pred_test = t.predict(X_test)
        y_pred_train_list.append(y_pred_train)
        y_pred_test_list.append(y_pred_test)
    y_pred_train_list = np.array(y_pred_train_list)
    y_pred_test_list = np.array(y_pred_test_list)
    # majority vote auf y_pred und y_test  # stats.mode
    return y_pred_train_list, y_pred_test_list


def majority_vote(y_pred_train_list, y_pred_test_list, y_train, y_test):
    """Führt Majority Vote durch und gibt Accuracy aus."""
    mv_train_pred = stats.mode(y_pred_train_list, axis=0).mode
    mv_test_pred = stats.mode(y_pred_test_list, axis=0).mode
    accuracy_train = accuracy_score(y_train, mv_train_pred[0])
    accuracy_test = accuracy_score(y_test, mv_test_pred[0])
    print("Accuracy train: ", accuracy_train)
    print("Accuracy test: ", accuracy_test)


# Daten vobereiten
iris = load_iris()
X = iris.data
X = X[:, [0, 1]]
y = iris.target
data = np.column_stack((X, y))

# Funktion aufrufen
random_forest(data, 100, 5)
