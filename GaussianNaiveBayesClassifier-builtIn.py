from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie zbioru danych Iris
iris = load_iris()

# Utworzenie DataFrame z danych
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['Species'] = iris.target

# Zamiana wartości liczbowych gatunków na nazwy
iris_data['Species'] = iris_data['Species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})

# Sprawdź dane
print(iris_data.head())
print(iris_data['Species'].value_counts())  # Po 50 próbek dla każdego gatunku


# Podział na gatunki
species_groups = iris_data.groupby('Species')

# Listy na dane treningowe i testowe
train_data = []
test_data = []

# Podział dla każdego gatunku
for species, group in species_groups:
    train, test = train_test_split(group, test_size=15, random_state=42)  # 15 testowych
    train_data.append(train)
    test_data.append(test)

# Połączenie danych w jeden DataFrame
train_data = pd.concat(train_data).reset_index(drop=True)
test_data = pd.concat(test_data).reset_index(drop=True)

# Sprawdzenie podziału
print("Rozmiar zestawu treningowego:", len(train_data))
print("Rozmiar zestawu testowego:", len(test_data))




# Rozdzielenie cech i etykiet
X_train = train_data.drop(columns='Species')
y_train = train_data['Species']

X_test = test_data.drop(columns='Species')
y_test = test_data['Species']

# Trenowanie modelu
model = GaussianNB()
model.fit(X_train, y_train)

# Przewidywanie
y_pred = model.predict(X_test)

# Ocena modelu
print(f"Dokładność: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))
