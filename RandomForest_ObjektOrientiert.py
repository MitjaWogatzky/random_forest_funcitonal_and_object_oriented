import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy import stats


class RandomForest:
    """Eine Klasse zum ausführen eines Random Forest."""
    # Klasse selber hat keine Argumente (nur zum Vererben), sondern nur der Konstruktor (init)

    def __init__(self, data, trees, max_depth):
        """Initialisiert die Daten und die Hyperparameter"""
        self.data = data
        self.trees = trees
        self.max_depth = max_depth

    def random_forest(self):
        """Random forest process."""
        data_premuted = np.random.permutation(self.data) # falls Daten nach y sortiert abgespeichert sind
        X_train, y_train, X_test, y_test = self.split_data(data_premuted)
        fittet_trees = self.fit(X_train, y_train, self.trees, self.max_depth)
        y_pred_train_list, y_pred_test_list = self.predict(fittet_trees,X_train, y_train, X_test, y_test)
        self.majority_vote(y_pred_train_list, y_pred_test_list, y_train, y_test)

    def split_data(self, test_size=0.33):
        """Splits data in train and test data."""
        z = int(len(self.data)*test_size)
        X_train = self.data[z:, 0:2]
        y_train = self.data[z:, 2]
        X_test = self.data[:z, 0:2]
        y_test = self.data[:z, 2]
        return X_train, y_train, X_test, y_test

    def permut(self, X_train, y_train):
        """Permutiert die Trainingsdaten."""
        z = int(len(X_train))
        perm = np.random.permutation(z)
        X_train_permuted = X_train[perm, :]
        y_train_permuted = y_train[perm,]
        return X_train_permuted, y_train_permuted

    def fit(self, X_train, y_train):
        """Trainiert die Trainingsdaten mit der Anzahl "trees" und der Tiefe "max_depth"."""
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        fittet_trees = []
        for t in range(self.trees):
            # X_train_perm, X_test_perm, y_train_perm, y_test_perm = permut(data)
            X_train_permuted, y_train_permuted = self.permut(X_train, y_train)
            z = int(len(X_train_permuted)/2)
            X_sub_train = X_train_permuted[:z, [0, 1]]
            y_sub_train = y_train_permuted[:z]
            fit_ = tree.fit(X_sub_train, y_sub_train)
            fittet_trees.append(fit_)
        return fittet_trees

    def predict(self, fittet_trees, X_train, y_train, X_test, y_test):
        """Macht die Vorhersage und gibt y predicted für Trainings- und Testdaten wieder."""
        y_pred_train_list, y_pred_test_list = [], []
        for t in fittet_trees:
            #X_train_perm, y_train_perm = permut(X_train, y_train)
            y_pred_train = t.predict(X_train)
            y_pred_test = t.predict(X_test)
            y_pred_train_list.append(y_pred_train)
            y_pred_test_list.append(y_pred_test)
        y_pred_train_list = np.array(y_pred_train_list)
        y_pred_test_list = np.array(y_pred_test_list)
        # majority vote auf y_pred und y_test  # stats.mode
        return y_pred_train_list, y_pred_test_list

    def majority_vote(self, y_pred_train_list, y_pred_test_list, y_train, y_test):
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
rf = RandomForest(data, 100, 5)
rf.random_forest()