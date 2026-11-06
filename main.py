from pathlib import Path

from objects.data import prepare_dataset
from objects.models.KNN import KNN
from objects.data import prepare_dataset
from objects.models.ANN import NeuralNetwork
from objects.models.naiveBayes import NaiveBayes

DATASETS = {
    "Breast Cancer": Path("./data/breast_cancer.csv"),
    "Mnist": Path("./data/mnist_train.csv"),
}


def main():
    for name, path in DATASETS.items():
        dataset = prepare_dataset(path, name)
        NaiveBayes.run(dataset)
        KNN.run(dataset)
        NeuralNetwork.run(dataset)


if __name__ == "__main__":
    main()
