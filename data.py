class Data:
    def __init__(self, features: list[float], label: str):
        self.features = features
        self.label = label

    def __str__(self):
        return f"Features: {self.features}, Label: {self.label}"


if __name__ == "__main__":
    features_example = [1.2, 3.4, 2.5]
    label_example = "catégorie A"
    data_instance = Data(features_example, label_example)

    print(data_instance)
