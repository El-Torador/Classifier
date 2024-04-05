import pandas as pd
from KNNClassifier import KNNClassifier
from EuclideanDistance import EuclideanDistance
from ManhattanDistance import ManhattanDistance
from ChebitchevDistance import ChebyshevDistance
from data import Data
from Distance import Distance
from csvLoader import load_csv
from splitDataset import split_dataset


def evaluate_knn(k: int, train_data: list[Data], test_data: list[Data], distance: Distance = EuclideanDistance()):
    # Initialisation du classifieur KNN avec EuclideanDistance
    knn_classifier = KNNClassifier(k=k, distance=distance)

    # Entraînement du classifieur avec les données d'entraînement
    knn_classifier.train(train_data)

    # Prédiction des classes pour les données de test
    predictions = knn_classifier.evaluate(test_data)

    # Calcul du pourcentage de bonnes réponses
    correct_predictions = sum(1 for i in range(len(test_data)) if predictions[i] == test_data[i].label)
    accuracy = correct_predictions / len(test_data) * 100

    return accuracy

k_values = [1, 10, 30]
training_data_file = "./datasets/iris.csv"
train2_data_file = "./datasets/bezdekIris.csv"

X_train, y_train = split_dataset(load_csv(training_data_file), 80)
X_test, y_test = split_dataset(load_csv(training_data_file), 20)


def test():
    # Liste des distances à tester
    distances = {
        "Manhattan": ManhattanDistance(),
        "Chebyshev": ChebyshevDistance(),
        "Euclidean": EuclideanDistance()
    }

    evaluations = {"Distance": [], "K": [], "Accuracy": []}

    for distance_name, distance_obj in distances.items():
        for k in k_values:
            accuracy = evaluate_knn(k, X_train, y_test, distance_obj)
            evaluations["Distance"].append(distance_name)
            evaluations["K"].append(k)
            evaluations["Accuracy"].append(accuracy)

    evaluations_df = pd.DataFrame(evaluations)

    print(evaluations_df)

    # Enregistrement des évaluations dans un fichier CSV
    # evaluations_df.to_csv("knn_evaluations_other_distances.csv", index=False)

test()