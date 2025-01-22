from threading import Thread
from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 42

def get_first_10_files() -> list:
    files = os.listdir('datasets')
    return files[:10]

def train(df: pd.DataFrame, dataset:str) -> list:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    rf = RandomForestClassifier(random_state=SEED)
    svm = SVC(random_state=SEED)
    results = []
    rkf = RepeatedKFold(n_repeats=2, n_splits=5)
    for fold, (train_index, test_index) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred)
        results.append(f"{dataset}, fold {fold + 1}, RandomForest, accuracy {accuracy_rf:.4f}")
        
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred)
        results.append(f"{dataset}, fold {fold + 1}, SVM, accuracy {accuracy_svm:.4f}")
        
    return results

def load_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df

def loop(file, output):
    df = load_file(f'datasets/{file}')
    accs = train(df, file)
    output.append(accs)

def main():
    start = datetime.now()
    files = get_first_10_files()

    threads = []
    output = []

    for file in files:
        thread = Thread(target=loop, args=(file, output))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    output = np.array(output).flatten()
    print(output)
    print(datetime.now() - start)

if __name__ == "__main__":
    main()
