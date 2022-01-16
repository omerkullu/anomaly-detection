import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split

liste = ["other/", "valve1/", "valve2/"]
all_df = pd.DataFrame()

for i in liste:
    for j in os.listdir(i):
        df = pd.read_csv(i+j,sep=";", index_col='datetime')
        all_df = pd.concat([all_df, df])



"""all_df["Temperature_f"] = 1.8 * all_df["Temperature"] + 32
all_df["Thermocouple_f"] = 1.8 * all_df["Thermocouple"] + 32
all_df["watts"] = all_df["Voltage"] * all_df["Current"]"""
        
"""import seaborn as sns
sns.heatmap(all_df.corr(), annot=True)"""


scaler_temp = MinMaxScaler()
all_df[["Temperature"]] = scaler_temp.fit_transform(all_df[["Temperature"]])

scaler_therm = MinMaxScaler()
all_df[['Thermocouple']] = scaler_therm.fit_transform(all_df[['Thermocouple']])

scaler_volt = MinMaxScaler()
all_df[['Voltage']] = scaler_volt.fit_transform(all_df[['Voltage']])

scaler_vol = MinMaxScaler()
all_df[['Volume Flow RateRMS']] = scaler_vol.fit_transform(all_df[['Volume Flow RateRMS']])

"""scaler_watts = MinMaxScaler()
all_df[['watts']] = scaler_watts.fit_transform(all_df[['watts']])"""

Y = all_df['anomaly']
X = all_df.drop(['anomaly', 'changepoint'], axis = 1)


"""x_train, x_test = X[:27750], X[27750:37000]
y_train, y_test = Y[:27750], Y[27750:37000]
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train, y_train, random_state=2)
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=42, shuffle=False)


from sklearn.neighbors import KNeighborsClassifier

print("KNN")
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

params = {
        'n_neighbors': [3,5,7],
        'metric': ['manhattan', 'minkowski', 'euclidean'],
    }

#svm.fit(x_train,y_train)
clf = GridSearchCV(estimator=knn, param_grid=params, scoring='accuracy', return_train_score=True, verbose=1, cv=3)
clf.fit(x_train, y_train.values.ravel())
print(clf.best_params_)

#knn.fit(x_train,y_train)
pred = clf.predict(x_test)

from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score, roc_auc_score

print(confusion_matrix(y_test, pred))
print(f1_score(y_test, pred))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(accuracy_score(y_test, pred))
print(roc_auc_score(y_test, pred))



print("XGBOOST")
from xgboost import XGBClassifier
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

best = {'colsample_bytree': 0.6,
 'gamma': 0.5,
 'max_depth': 3,
 'min_child_weight': 10,
 'subsample': 0.6}

xgb = XGBClassifier()
clf = GridSearchCV(estimator=xgb, param_grid=params, scoring='accuracy', return_train_score=True, verbose=1, cv=3)
clf.fit(x_train, y_train.values.ravel())
print(clf.best_params_)
#xgb.fit(x_train,y_train)
pred = clf.predict(x_test)


print(confusion_matrix(y_test, pred))
print(f1_score(y_test, pred))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(accuracy_score(y_test, pred))
print(roc_auc_score(y_test, pred))

