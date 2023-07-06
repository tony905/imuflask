#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv("Final_dataset.csv")

from sklearn.model_selection import train_test_split

train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

X_train = train_df.drop('Label', axis=1)
y_train = train_df['Label']

X_val = val_df.drop('Label', axis=1)
y_val = val_df['Label']

X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaledS = scaler.fit_transform(X_train)
X_val_scaledS = scaler.transform(X_val)
X_test_scaledS = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn_model1 = KNeighborsClassifier()
knn_model1.fit(X_train_scaledS, y_train)
knn_accuracy_train = knn_model1.score(X_train_scaledS, y_train)
knn_accuracy_val = knn_model1.score(X_val_scaledS, y_val)
knn_accuracy_test = knn_model1.score(X_test_scaledS, y_test)


pickle.dump(knn_model1,open('knn.pkl','wb'))
model=pickle.load(open('knn.pkl','rb'))
