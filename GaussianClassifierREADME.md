# GaussianNaiveBayesClassifier

Implementacja klasyfikatora oparta na założeniu niezależności cech oraz rozkładzie Gaussa. Klasyfikator został zaprojektowany do klasyfikacji danych ciągłych.
Poniżej opisuję wersję programu, które nie wykorzystuje wbudowanych bibliotek dla funkcji fit, predict oraz predict_proba. Wykorzystałem jednak wbudowane narzędzia w osobnym pliku w celu lepszej analizy wyników niż sama dokładność.

## Funkcjonalności

- **Trenowanie modelu** (`fit`): Dopasowanie klasyfikatora do danych treningowych.
- **Przewidywanie klas** (`predict`): Przewidywanie etykiet dla nowych danych.
- **Prawdopodobieństwo klas** (`predict_proba`): Obliczanie prawdopodobieństw przynależności danych do klas.

## Wymagania

Do uruchomienia projektu potrzebne są:
- Python 3.7 lub nowszy
- NumPy
- scikit-learn (do testowania i porównywania)

Zainstaluj wymagane pakiety za pomocą:
```bash
pip install numpy scikit-learn
```

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


## Analiza wyników programu

### **1. Charakterystyka zbioru danych**
Zbiór danych Iris zawiera:
- **150 próbek** podzielonych równomiernie pomiędzy trzy gatunki:
  - `Iris-setosa`: 50 próbek,
  - `Iris-versicolor`: 50 próbek,
  - `Iris-virginica`: 50 próbek.
- Każda próbka jest opisana przez 4 cechy:
  - **Sepal Length (cm)**: Długość działki kielicha,
  - **Sepal Width (cm)**: Szerokość działki kielicha,
  - **Petal Length (cm)**: Długość płatka,
  - **Petal Width (cm)**: Szerokość płatka.


### **2. Podział na zestawy treningowe i testowe**
Dane zostały podzielone na:
- **Zestaw treningowy**: 105 próbek (po 35 próbek z każdego gatunku),
- **Zestaw testowy**: 45 próbek (po 15 próbek z każdego gatunku).


### **3. Wyniki klasyfikatora**
Model osiągnął bardzo wysoką skuteczność:
- **Dokładność ogólna**: **97.78%** – model poprawnie sklasyfikował 44 z 45 próbek.


### **4. Szczegółowa analiza wyników**

#### **Raport klasyfikacji**:
- **Precision**: Procent próbek przewidzianych poprawnie dla danej klasy:
  - `Iris-setosa`: **100%**
  - `Iris-versicolor`: **94%**
  - `Iris-virginica`: **100%**
- **Recall**: Procent próbek danej klasy poprawnie przewidzianych:
  - `Iris-setosa`: **100%**
  - `Iris-versicolor`: **100%**
  - `Iris-virginica`: **93%**
- **F1-score**: Harmoniczna średnia `precision` i `recall`:
  - `Iris-setosa`: **1.00**
  - `Iris-versicolor`: **0.97**
  - `Iris-virginica`: **0.97**

#### **Macierz błędów**:
Model błędnie zaklasyfikował jedną próbkę:
- Jedna próbka `Iris-virginica` została błędnie przypisana do `Iris-versicolor`. Po analizie danych z wykorzystaniem box plotów oraz kwartyli jestem w stanie stwierdzić wartość odstającą dla cechy Sepal Lenght próbki o Id 107 i gatunku Iris-virginica.


### **5. Wnioski**
- Model Gaussian Naive Bayes dobrze radzi sobie z klasyfikacją danych Iris, osiągając wysoką dokładność.
- **`Iris-setosa`** została sklasyfikowana bezbłędnie dzięki wyraźnemu odseparowaniu tej klasy od pozostałych.
- Większość błędów klasyfikacji wynika z nakładania się cech między klasami `Iris-versicolor` i `Iris-virginica`.

