import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None  # Unikalne klasy w danych
        self.priors = {}     # Prior dla każdej klasy
        self.likelihoods = {}  # Parametry rozkładu Gaussa (mean, var)

    def fit(self, X, y):
        """
        Trenuje model na danych.
        X: ndarray, cechy
        y: ndarray, etykiety
        """
        self.classes = np.unique(y)
        self.priors = {}
        self.likelihoods = {}

        for c in self.classes:
            # Wybierz próbki dla klasy c
            X_c = X[y == c]
            # Oblicz prior
            self.priors[c] = len(X_c) / len(X)
            # Oblicz parametry rozkładu normalnego (średnia i wariancja) dla każdej cechy
            self.likelihoods[c] = {
                "mean": np.mean(X_c, axis=0),
                "var": np.var(X_c, axis=0)
            }

    def _gaussian_likelihood(self, mean, var, x):
        """
        Oblicza likelihood (gęstość rozkładu Gaussa).
        mean: ndarray, średnia
        var: ndarray, wariancja
        x: ndarray, dane
        """
        eps = 1e-6  # Drobna wartość dla stabilności
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coeff * exponent

    def predict_proba(self, X):
        """
        Zwraca prawdopodobieństwa przynależności dla każdej klasy.
        X: ndarray, dane wejściowe
        """
        probs = np.zeros((X.shape[0], len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            prior = self.priors[c]
            mean = self.likelihoods[c]["mean"]
            var = self.likelihoods[c]["var"]
            likelihood = self._gaussian_likelihood(mean, var, X)
            probs[:, idx] = np.log(prior) + np.sum(np.log(likelihood), axis=1)

        # Normalizacja prawdopodobieństw (log-sum-exp dla stabilności numerycznej)
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Przewiduje klasy dla danych.
        X: ndarray, dane wejściowe
        """
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


# Generowanie danych Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Wczytaj dane
iris = load_iris()
X = iris.data
y = iris.target

# Podział na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie własnego klasyfikatora
clf = NaiveBayesClassifier()
clf.fit(X_train, y_train)

# Przewidywanie
y_pred = clf.predict(X_test)

# Ocena modelu
from sklearn.metrics import accuracy_score
print(f"Dokładność: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
