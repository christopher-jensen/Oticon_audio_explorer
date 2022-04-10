from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import load_data as ld
import numpy as np

def transform_data(df):
    x = df.iloc[:,-2].values
    x = x.reshape(-1,1)
    y = df.iloc[:,-1:].values
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, n_of_trees):
    classifer = RandomForestClassifier(n_estimators=n_of_trees, random_state=0, max_depth=10)
    classifer.fit(x_train, np.ravel(y_train))
    return classifer

def predict(classifier, x_test):
    y_pred = classifier.predict(x_test)
    return y_pred

def print_random_forest_predictions(y_test, y_pred, n_of_trees, depth):
    print(f"----------------------- {n_of_trees} trees -----------------------")
    print(f"----------------------- {depth} depth -----------------------")
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print(conf_matrix)
    # Uncomment if visualisation of confusion matrix is wanted
    # plt.show()
    precision = classification_report(y_test,y_pred)
    print(precision)

    # Uncomment if you want results seperated by type
    # precision,recall,fscore,support=score(y_test,y_pred,average=None)

    acc_score = accuracy_score(y_test, y_pred)
    print(acc_score)
    print()
    print()


def main():
    data = ld.get_data_with_labels()
    x_train, x_test, y_train, y_test = transform_data(data)
    for i in range(1,2):
        for j in range(1, 2):
            # n_of_trees = 1*10**i
            n_of_trees = i
            depth = j
            classifier = train_model(x_train, y_train, n_of_trees)
            y_pred = classifier. predict(x_test)
            print_random_forest_predictions(y_test, y_pred, n_of_trees, depth)

if __name__ == "__main__":
    main()