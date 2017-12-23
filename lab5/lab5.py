import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# загрузка датасета
def load_data(filename):
    dataset_by_string = pd.read_csv(filename, header=None).values[1:]
    dataset = []
    for i in range(len(dataset_by_string)):
        dataset.append(dataset_by_string[i][0].split(",")[2:])
    return np.array(dataset)

# разделение датасета на признаки и метки классов
def split_data():
    dataset = load_data('datatest.csv')
    occ_attr = dataset[:,:-1]
    occ_class = dataset[:,-1]
    occ_class = occ_class.astype(np.float)
    occ_attr = occ_attr.astype(np.float)
    return occ_attr, occ_class

# расчет и отображение метрик
def get_metrics(site_attr, site_class, model, k_fold, scoring, modelName):
    result = cross_val_score(model, site_attr, site_class, cv=k_fold, scoring=scoring)
    print(" %s: %0.3f (%0.3f)" % (modelName, result.mean(), result.std() ))

def get_confusion_matrix(data_train, data_test, class_train, class_test, model):
    model.fit(data_train, class_train)
    model_predicted = model.predict(data_test)
    model_matrix = confusion_matrix(class_test, model_predicted)
    print(model_matrix)
    return model_predicted

def get_classification_report(model_predicted, class_test):
    model_r = classification_report(class_test, model_predicted)
    print(model_r)

def main():
    X, Y = split_data()
    data_train, data_test, class_train, class_test = train_test_split(X, Y, test_size=0.30, random_state=123)
    kFold = KFold(n_splits=150, random_state=50, shuffle=True)

    model1 = LogisticRegression()
    model2 = QDA()

    print("Accuracy:")
    get_metrics(X, Y, model1, kFold, 'accuracy', "Logistic Regression")
    get_metrics(X, Y, model2, kFold, 'accuracy', "Quadratic Discriminant Analysis")

    print("Logarithmic Loss:")
    get_metrics(X, Y, model1, kFold, 'neg_log_loss', "Logistic Regression")
    get_metrics(X, Y, model2, kFold, 'neg_log_loss', "Quadratic Discriminant Analysis")

    print("Area Under ROC Curve:")
    get_metrics(X, Y, model1, kFold, None, "Logistic Regression")
    get_metrics(X, Y, model2, kFold, None, "Quadratic Discriminant Analysis")

    print("Confusion Matrices:")
    print(" - Logistic Regression:")
    model1_predicted = get_confusion_matrix(data_train, data_test, class_train, class_test, model1)
    print(" - Quadratic Discriminant Analysis:")
    model2_predicted = get_confusion_matrix(data_train, data_test, class_train, class_test, model2)

    print("Classification Reports:")
    print(" - Logistic Regression:")
    get_classification_report(model1_predicted, class_test)
    print(" - Quadratic Discriminant Analysis:")
    get_classification_report(model2_predicted, class_test)

main()
