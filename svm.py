import os
import pandas as pd
from dotenv import load_dotenv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import flatten_data as fd
import load_data as ld

class SVCConfiguration:
    def __init__(self, kernel, soft_margin, gamma, poly_degrees=2):
        self.kernel = kernel
        self.soft_margin = soft_margin
        self.gamma = gamma
        self.poly_degrees = poly_degrees
        self.accuracy=None
        self.report=None

    def __str__(self):
        poly_degrees = f', poly_degrees:{self.poly_degrees}' if self.kernel == 'poly' else ''
        return f'{self.accuracy}: <kernel:{self.kernel}, soft_margin:{self.soft_margin}, gamma:{self.gamma}{poly_degrees}>'

    def set_accuracy(self, accuracy: float):
        self.accuracy = accuracy

    def set_report(self, report):
        self.report = report

def create_and_fit_SVC_classifier(X_train, Y_train) -> SVCConfiguration:
    classifier = svm.SVC(kernel="rbf")
    classifier.fit(X_train, Y_train.iloc[:,-1])
    return classifier

def get_train_test_split(df: pd.DataFrame):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1:]
    return train_test_split( X, Y, test_size=0.25)

def main():
    df = fd.flatten_dataframe_to_n2_frequency_band()
    df_test = fd.flatten_dataframe_to_n2_frequency_band_unlabeled()
    # df = ld.get_data_with_labels()
    # df_test = ld.get_test_data()

    X_train, X_test, Y_train, Y_test = get_train_test_split(df)

    # Create and fit model   
    # model_config = get_default_config()
    classifier   = create_and_fit_SVC_classifier(X_train, Y_train)

    # Get predictions and measure accuracy
    Y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test.iloc[:,-1], Y_pred)
    report = classification_report(Y_test.iloc[:,-1], Y_pred)
    print(accuracy)
    print("------------------------------------------------------------")
    print(report)

if __name__ == '__main__':
    main()