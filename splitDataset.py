import random
from data import Data

def split_dataset(dataset: list[Data], percentage: float):
    # Calcul du nombre d'échantillons à prendre pour le premier sous-ensemble
    num_samples = len(dataset)
    num_first_subset = int(num_samples * percentage / 100.0)

    # Mélanger aléatoirement les données
    random.shuffle(dataset)

    # Séparation en deux sous-ensembles
    first_subset = dataset[:num_first_subset]
    second_subset = dataset[num_first_subset:]

    return first_subset, second_subset


if __name__ == "__main__":
    data_vector = []
    # Supposons que vous avez un vecteur de données nommé "data_vector"
    percentage = 20  # Pourcentage de données à mettre dans le premier sous-ensemble
    first_subset, second_subset = split_dataset(data_vector, percentage)

    print("Premier sous-ensemble:")
    for data_instance in first_subset:
        print(data_instance)

    print("\nDeuxième sous-ensemble:")
    for data_instance in second_subset:
        print(data_instance)
