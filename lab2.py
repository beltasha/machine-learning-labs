import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
	data_train, data_test, class_train, class_test = train_test_split(occ_attr, occ_class, test_size=test_size, random_state=55)
	return data_train, class_train, data_test, class_test

def main():
    max_size = 0.4
    min_size = step = 0.1

    for size in np.arange(min_size, max_size, step):
        data_train, class_train, data_test, class_test = split_dataset(size)
        desisionForest = DecisionTreeClassifier()
        desisionForest = desisionForest.fit(data_train, class_train)
        desisionAccuracy = desisionForest.score(data_test, class_test)
        print("DecisionTree test size: ", size, 'Accuracy: ', desisionAccuracy)

        randonForest = RandomForestClassifier()
        randonForest = randonForest.fit(data_train, class_train)
        randomAccuracy = randonForest.score(data_test, class_test)
        print("RandomTree test size: ", size, 'Accuracy: ', randomAccuracy)

main()