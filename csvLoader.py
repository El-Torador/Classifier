import csv
from data import Data

def load_csv(filename: str) -> list[Data]:
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # On suppose que les caractéristiques sont dans les colonnes 0 à n-2 et la classe dans la dernière colonne
            features = [float(x) for x in row[:-1]]
            label = row[-1] if len(features) else None
            if label is None:
                continue
            data_instance = Data(features, label)
            dataset.append(data_instance)
    return dataset


if __name__ == "__main__":
    filename = "./datasets/iris.csv"
    data_vector = load_csv(filename)
    for data_instance in data_vector:
        print(data_instance)