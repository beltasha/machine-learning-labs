import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import metrics

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

# признаки и соответствующее значение класса(нахождение в комнате)
def data_description(occ_attr, occ_class):
    columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Class']
    data = pd.DataFrame(load_data('datatest.csv'), columns = columns)
    print(data.head())

    print('Number of records:', occ_class.shape[0])
    print('Number of signs:', occ_attr.shape[1])

    print('\nThe shares of each of the classes')
    print('Class 0 (Nobody in room): {:.2%}'.format(list(occ_class).count(0)/occ_class.shape[0]))
    print('Class 1 (Smbdy in room): {:.2%}'.format(list(occ_class).count(1)/occ_class.shape[0]))

def data_2D_visualization(occ_attr, occ_class):
    plt.figure(figsize=(6,5))
    for label,marker,color in zip(
        range(0,2),('x', 'o'),('red', 'green')):
        
        # Вычисление коэффициента корреляции Пирсона
        R = pearsonr(occ_attr[:,0][occ_class == label], occ_attr[:,1][occ_class == label])
        plt.scatter(x=occ_attr[:,0][occ_class == label],
                y=occ_attr[:,1][occ_class == label],
                marker=marker,
                color=color,
                alpha=0.7, 
                label='class {:}, R={:.2f}'.format(label, R[0])
                )
    
    plt.title('Occupancy Detection Data Set')
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    plt.legend(loc='upper right')
    plt.show()

def data_3D_visualization(occ_attr, occ_class):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    for label,marker,color in zip(
        range(0,2),('x', 'o'),('red', 'green')):

        # Вычисление коэффициента корреляции Пирсона
        ax.scatter(occ_attr[:,0][occ_class == label],
                    occ_attr[:,1][occ_class == label],
                    occ_attr[:,2][occ_class == label],
                marker=marker,
                color=color,
                s=40,
                alpha=0.7, 
                label='class {:}'.format(label)
                )
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_zlabel('Light')

    plt.title('Occupancy Detection Data Set')
    plt.legend(loc='upper right')
    plt.show()

def train_test_visualization(data_train, data_test, class_train, class_test):
    std_scale = preprocessing.StandardScaler().fit(data_train)
    data_train = std_scale.transform(data_train)
    data_test = std_scale.transform(data_test)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

    for a,x_dat, y_lab in zip(ax, (data_train, data_test), (class_train, class_test)):

        for label,marker,color in zip(
            range(0,2),('x', 'o'),('red','green')):

            a.scatter(x=x_dat[:,0][y_lab == label], 
                    y=x_dat[:,1][y_lab == label], 
                    marker=marker, 
                    color=color,   
                    alpha=0.7,   
                    label='class {}'.format(label)
                    )

        a.legend(loc='upper right')

    ax[0].set_title('Training Dataset')
    ax[1].set_title('Test Dataset')
    f.text(0.5, 0.04, 'Temperature (standardized)', ha='center', va='center')
    f.text(0.08, 0.5, 'Humidity (standardized)', ha='center', va='center', rotation='vertical')

    #plt.show()

def train_data(occ_attr, occ_class):
    data_train, data_test, class_train, class_test = train_test_split(occ_attr, occ_class, test_size=0.30, random_state=123)
    print('\nThe shares of each of the classes')
    print('\nTraining Dataset:')
    print('Class 0 (Nobody in room): {:.2%}'.format(list(class_train).count(0)/class_train.shape[0]))
    print('Class 1 (Smbdy in room): {:.2%}'.format(list(class_train).count(1)/class_train.shape[0]))

    print('\nTest Dataset:')
    print('Class 0 (Nobody in room): {:.2%}'.format(list(class_test).count(0)/class_test.shape[0]))
    print('Class 1 (Smbdy in room): {:.2%}'.format(list(class_test).count(1)/class_test.shape[0]))

    train_test_visualization(data_train, data_test, class_train, class_test)
    return data_train, data_test, class_train, class_test

def linear_discriminant_analysis(data_train, class_train):
    sklearn_lda = LDA()
    sklearn_transf = sklearn_lda.fit(data_train, class_train).transform(data_train)

    plt.figure(figsize=(8,8))
    for label,marker,color in zip(
        range(0,2),('x', 'o'),('red', 'green')):

        plt.scatter(x=sklearn_transf[class_train == label],
                    y=sklearn_transf[class_train == label], 
                    marker=marker, 
                    color=color,
                    alpha=0.7, 
                    label='class {}'.format(label))

    plt.xlabel('vector 1')
    plt.ylabel('vector 2')

    plt.legend()
    # Визуализация разбиения классов после линейного преобразования LDA
    plt.title('Most significant singular vectors after linear transformation via LDA')

    plt.show()

def train_linear_discriminant_analysis(data_train, data_test, class_train, class_test):
    lda_clf = LDA()
    lda_clf.fit(data_train, class_train)
    #LDA(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)

    pred_train = lda_clf.predict(data_train)
    print('Linear discriminant analysis')
    print('The accuracy of the classification on the training set of data')
    print('{:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))

    pred_test = lda_clf.predict(data_test)

    print('The accuracy of classification on the test data set')
    print('{:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))

def train_quadratic_discriminant_analysis(data_train, data_test, class_train, class_test):
    qda_clf = QDA()
    qda_clf.fit(data_train, class_train)

    pred_train = qda_clf.predict(data_train)
    print('Quadratic discriminant analysis')
    print('The accuracy of the classification on the training set of data')
    print('{:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))

    pred_test = qda_clf.predict(data_test)

    print('The accuracy of classification on the test data set')
    print('{:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))

def main():
    occ_attr, occ_class = split_data()
   
    data_description(occ_attr, occ_class)
    data_2D_visualization(occ_attr, occ_class)
    data_3D_visualization(occ_attr, occ_class)
    
    data_train, data_test, class_train, class_test = train_data(occ_attr, occ_class)
    
    linear_discriminant_analysis(data_train, class_train)
    train_linear_discriminant_analysis(data_train, data_test, class_train, class_test)
    train_quadratic_discriminant_analysis(data_train, data_test, class_train, class_test)

main()



