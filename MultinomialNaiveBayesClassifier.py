class Trait:
    def __init__(self, name: str, classes: list[str], possible_values: list[str]):
        """Początkowo self.probabilities będzie liczyło, ile jest grzybów z danej klasy (trujący/jadalny) o danej wartości
        cechy. Liczenie zaczynamy od 1, aby każda wartość cechy miała niezerowe prawdopodobieństwo (coś jak dodawanie czarnego
        elementu w nagraniu ze StatQuest, chyba nazywa się to "Laplace smoothing").
        Później, po wykonaniu funkcji calculate_probabilities self.probabilities[cls][trait_value] zwraca
        prawdopodobieństwo należenia grzyba do klasy cls, przy założeniu, że grzyb wykazuje cechę o wartości trait_value."""
        self.name = name
        self.classes = classes
        self.probabilities = {cls: {value: 1 for value in possible_values} for cls in classes}


    def calculate_probabilities(self):
        """Funkcja zamienia self.probabilities ze słownika liczącego, ile było grzybów danej klasy o danej wartości cechy,
        na słownik prawdopodobieństw"""
        for cls in self.classes:
            sum_of_counts = sum(self.probabilities[cls].values())
            for trait_value in self.probabilities[cls].keys():
                self.probabilities[cls][trait_value] /= sum_of_counts


class MultinomialNaiveBayesClassifier:
    def __init__(self, classes: list[str], traits: list[Trait]):
        self.classes = classes

        self.traits = traits

        self.class_probabilities = {cls: 1 for cls in classes}
        # p(c_i), prawdopodobieństwo przynależności do danej klasy (a priori)

    def calculate_class_probabilities(self):
        """Funkcja podobna do Trait.calculate_probabilities - zamienia class_probabilities ze słownika zliczającego
        wystąpienia poszczególnych klas na słownik prawdopodobieństw"""
        sum_of_counts = sum(self.class_probabilities.values())
        self.class_probabilities = {cls: value / sum_of_counts for cls, value in self.class_probabilities.items()}

    def fit(self, training_set: list[list[str]]):
        """Funkcja służąca do trenowania klasyfikatora na przykładowych danych"""
        self.__reset_probabilities__()
        for mushroom in training_set:
            current_mushroom_class = mushroom[0]
            self.class_probabilities[current_mushroom_class] += 1
            for i, trait_value in enumerate(mushroom[1:]):
                if trait_value == "?":
                    continue
                self.traits[i].probabilities[current_mushroom_class][trait_value] += 1

        self.calculate_class_probabilities()
        for trait in self.traits:
            trait.calculate_probabilities()

    def predict_proba(self, data_vector: list[str]) -> dict[str, float]:
        """Dla danego wektora danych (jeden wektor to wypisane po kolei cechy jednego, konkretnego grzyba) funkcja zwraca
        prawdopodobieństwa przynależności do każdej klasy"""
        result = dict()
        for cls in self.classes:
            current_class_probability = self.class_probabilities[cls]
            for i, trait_value in enumerate(data_vector):
                if trait_value == '?':
                    continue
                current_class_probability *= self.traits[i].probabilities[cls][trait_value]
            result[cls] = current_class_probability
        return result

    def predict(self, data_vector: list[str]) -> str:
        """Funkcja zwraca klasę, do której prawdopodobnie należy dany grzyb (reprezentowany jako wektor danych)"""
        data_class_probabilities = self.predict_proba(data_vector)
        return max(data_class_probabilities, key=data_class_probabilities.get)

    def __reset_probabilities__(self) -> None:
        """Funkcja resetuje prawdopodobieństwa wszystkich cech i ustawia liczniki na 1 (tak, żeby można było najpierw zliczyć
        wystąpienia poszczególnych cech w czasie treningu, a potem obliczyć prawdopodobieństwo"""
        for cls in self.class_probabilities.keys():
            self.class_probabilities[cls] = 1
        for trait in self.traits:
            for cls in trait.probabilities.keys():
                for value in trait.probabilities[cls].keys():
                    trait.probabilities[cls][value] = 1


def set_up(data: list[list[str]]) -> MultinomialNaiveBayesClassifier:
    """Funkcja przygotowuje klasyfikator na podstawie danych z podanego pliku, ale go nie trenuje (w tym celu należy wywołać
    funkcję fit)"""
    classes = set()
    for i in range(1, len(data)):
        classes.add(data[i][0])

    traits = []
    for i in range(1, len(data[0])):
        trait_name = data[0][i]
        trait_values = set()
        for j in range(1, len(data)):
            if data[j][i] != '?':
                trait_values.add(data[j][i])
        traits.append(Trait(trait_name, list(classes), list(trait_values)))

    return MultinomialNaiveBayesClassifier(list(classes), traits)
