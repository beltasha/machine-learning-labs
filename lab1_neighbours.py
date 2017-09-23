from __future__ import division
import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import train_test_split
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# загрузка датасета
def load_data(filename):
	dataset_by_string = pd.read_csv(filename, header=None).values[1:]
	dataset = []
	for i in range(len(dataset_by_string)):
		dataset.append(dataset_by_string[i][0].split(","))
	return np.array(dataset)
	
# разделение датасета на тестовую и обучающую выборку
def split_dataset(test_size):
	dataset = load_data('datatest.csv')
	occ_attr = dataset[:, 2:-1]
	occ_class = dataset[:, -1]
	occ_class = occ_class.astype(np.float)
	occ_attr = occ_attr.astype(np.float)
	data_train, data_test, class_train, class_test = train_test_split(occ_attr, occ_class, test_size=test_size)
	return data_train, class_train, data_test, class_test

# евклидово расстояние от объекта №1 до объекта №2
def euclidean_distance(instance1, instance2):
    squares = [(i - j) ** 2 for i, j in zip(instance1, instance2)]
    return sqrt(sum(squares))

# рассчет расстояний до всех объектов в датасете
def get_neighbours(instance, data_train, class_train, k):
    distances = []
    for i in data_train:
        distances.append(euclidean_distance(instance, i))
    distances = tuple(zip(distances, class_train))
    # cортировка расстояний по возрастанию
    # k ближайших соседей
    return sorted(distances, key=operator.itemgetter(0))[:k]

# определение самого распространенного класса среди соседей
def get_response(neigbours):
    return Counter(neigbours).most_common()[0][0][1]

# классификация тестовой выборки
def get_predictions(data_train, class_train, data_test, k):
    predictions = []
    for i in data_test:
        neigbours = get_neighbours(i, data_train, class_train, k)
        response = get_response(neigbours)
        predictions.append(response)
    return predictions

# измерение точности
def get_accuracy(data_train, class_train, data_test, class_test, k):
    predictions = get_predictions(data_train, class_train, data_test, k)
    mean = [i == j for i, j in zip(class_test, predictions)]
    return sum(mean) / len(mean)

def main():
    data_train, class_train, data_test, class_test = split_dataset(0.35)
    print('myKNClass', 'Accuracy: ', get_accuracy(data_train, class_train, data_test, class_test, 15))

    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(data_train, class_train)
    print('sklKNClass', 'Accuracy: ', clf.score(data_test, class_test))

main()
