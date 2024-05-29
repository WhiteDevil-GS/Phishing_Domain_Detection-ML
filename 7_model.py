import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

phishing_data = pd.read_csv('original_new_phish_25k.csv', dtype=str, low_memory=False)
legitimate_data = pd.read_csv('legit_data.csv', dtype=str, low_memory=False)
phishing_data['Label'] = 1
legitimate_data['Label'] = 0
dataset = pd.concat([phishing_data, legitimate_data])
dataset = dataset.drop(['url', 'NonStdPort', 'GoogleIndex', 'double_slash_redirecting', 'https_token'], axis=1)
dataset = clean_data(dataset)
X = dataset.drop('Label', axis=1)
y = dataset['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Multilayer Perceptron': MLPClassifier(max_iter=300, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5) ,
    'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42)
}

best_accuracy = 0
best_model = None

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

model_filename = 'our_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved as {model_filename} with accuracy {best_accuracy:.4f}")
