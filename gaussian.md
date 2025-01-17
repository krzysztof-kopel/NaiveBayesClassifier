# NaiveBayesClassifier

`NaiveBayesClassifier` to implementacja klasyfikatora Naive Bayes oparta na założeniu niezależności cech oraz rozkładzie Gaussa (Gaussian Naive Bayes). Klasyfikator został zaprojektowany w celach edukacyjnych i może być używany do klasyfikacji danych ciągłych.

---

## Funkcjonalności

- **Trenowanie modelu** (`fit`): Dopasowanie klasyfikatora do danych treningowych.
- **Przewidywanie klas** (`predict`): Przewidywanie etykiet dla nowych danych.
- **Prawdopodobieństwo klas** (`predict_proba`): Obliczanie prawdopodobieństw przynależności danych do klas.

---

## Wymagania

Do uruchomienia projektu potrzebne są:
- Python 3.7 lub nowszy
- NumPy
- scikit-learn (do testowania i porównywania)

Zainstaluj wymagane pakiety za pomocą:
```bash
pip install numpy scikit-learn
```

---

## Jak używać

### 1. Import klasy i przygotowanie danych
Załaduj dane za pomocą np. `scikit-learn` i podziel je na zestawy treningowe oraz testowe:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from NaiveBayesClassifier import NaiveBayesClassifier

# Wczytaj dane
iris = load_iris()
X = iris.data
y = iris.target

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 2. Trenowanie modelu
Stwórz instancję klasy `NaiveBayesClassifier` i wytrenuj model na danych treningowych:
```python
clf = NaiveBayesClassifier()
clf.fit(X_train, y_train)
```

### 3. Przewidywanie klas
Użyj metody `predict` do przewidywania etykiet na danych testowych:
```python
y_pred = clf.predict(X_test)
```

### 4. Ocena modelu
Ocena wyników z użyciem `scikit-learn`:
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")
```

---

## Opis metod

### Konstruktor `__init__`
Inicjalizuje obiekt klasyfikatora i przygotowuje przestrzeń na priory oraz parametry rozkładu Gaussa.

### Metoda `fit`
- **Argumenty**:
  - `X` (ndarray): Dane treningowe (próbek, cech).
  - `y` (ndarray): Etykiety klas.
- **Opis**: Oblicza priory dla każdej klasy oraz średnią i wariancję dla cech.

### Metoda `predict`
- **Argumenty**:
  - `X` (ndarray): Dane wejściowe.
- **Opis**: Zwraca przewidywane klasy dla próbek w `X`.

### Metoda `predict_proba`
- **Argumenty**:
  - `X` (ndarray): Dane wejściowe.
- **Opis**: Oblicza prawdopodobieństwa przynależności do każdej klasy.

---

## Przykładowe dane wejściowe i wyjściowe

### Dane wejściowe
Przykład danych wejściowych w formacie NumPy array:
```python
X = [[5.1, 3.5, 1.4, 0.2],
     [4.9, 3.0, 1.4, 0.2]]
y = [0, 0]
```

### Dane wyjściowe
Po trenowaniu na danych treningowych:
```python
y_pred = clf.predict(X)
print(y_pred)  # Output: [0, 0]
```

---
