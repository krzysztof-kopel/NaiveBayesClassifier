# NaiveBayesClassifier
Naiwny klasyfikator bayesowski — projekt przygotowany w ramach przedmiotu "rachunek prawdopodobieństwa i statystyka" w semestrze zimowym roku akademickiego 2024/25.

## Autorzy:
- Gabriel Kania
- Krzysztof Kopel

## Klasyfikator do zmiennych kategorycznych:
Klasyfikator służy do klasyfikacji różnych gatunków grzybów na podstawie ich cech. W ramach tej części projektu przygotowane zostały pliki:

mushroom_data_analysis.xlsm -> zawiera analizę danych dotyczących grzybów
MultinomialNaiveBayesClassifier.py -> zawiera klasę opisującą klasyfikator dla zmiennych kategorycznych (w tym jego implementację), a także klasę opisującą cechę grzyba.
multinomial_classifier.ipynb -> notebook przedstawiający kolejne kroki działania klasyfikatora. Najlepiej korzystać z klasyfikatora właśnie przy użyciu notebooka.

Klasyfikator posiada 3 główne metody: 
fit(self, training_set) - metoda pozwalająca na trenowanie klasyfikatora, przyjmuje jako wejście zbiór danych, który ma zostać wykorzystany do trenowania.
predict_proba(self, data_vector) - zwraca prawdopodobieństwo przynależności danego grzyba (opisanego listą cech *data_vector*)
predict(self, data_vector) - zwraca klasę (w formie stringa) do której prawdopodobnie należy grzyb opisywany przez data_vector.
Dokładny opis opis znajduje się w komentarzach w kodzie w pliku MultinomialNaiveBayesClassifier.py.

## Klasyfikator do zmiennych ciągłych
Pełna dokumentacja w pliku **GaussianClassifierREADME.md**
