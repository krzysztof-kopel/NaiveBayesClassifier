import csv

def read_csv_file(file_path: str, keep_the_first_line: bool) -> list[list[str]]:
    """Funkcja wczytuje plik csv podany jako file_path i przekształca go w listę dwuwymiarową stringów."""
    result = []
    with open(file_path) as file:
        reader = csv.reader(file)

        for row in reader:
            result.append(row)

    return result if keep_the_first_line else result[1:]
