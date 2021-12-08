# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:59:51 2021

@author: 14076
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns 
sns.set(style="white")
sns.set(style="darkgrid", color_codes=True)

# import the data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
print(df.shape)
df.info()

# explore the data
# drop the first column "id"
data = df.drop('id', axis=1)
print(data.shape)

# check the missing values
round(df.isnull().sum()/df.shape[0]*100,2) #3.93% of BMI is missing

# impute the missing data of bmi with its mean
data.bmi.fillna(np.mean(data.bmi), inplace = True)

cat_data = [x for x in data.columns if 
            data[x].dtype == "object" or data[x].dtype =="int64"]
num_data = [y for y in data.columns if 
            data[y].dtype != "object" and data[y].dtype != "int64"]

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
for i in cat_data:
    print(i," : ", Counter(data[i]), '\n')


f, ax = plt.subplots(2,4,figsize=(24,10), sharey=False)
for i in range(len(cat_data)):
    sns.countplot(x=cat_data[i], data=data, ax=ax[i//4][i%4], 
                  palette=sns.cubehelix_palette(6, start = 0.7, rot = -.75))
plt.tight_layout()
plt.show()

# delete the observation with gender being "other"
data = data[data.gender!="Other"]
print(data.shape)

# create dummy variables for the categorical variables
data_dummies = pd.get_dummies(data, columns=cat_data, drop_first=True)
print(data_dummies.shape)
data_dummies.head()

import matplotlib.pyplot as plt
import matplotlib
# check the distribution of the numerical variables
f, axes = plt.subplots(1,len(num_data), figsize=(15,4), sharex=False)
for i in range(len(num_data)):
    sns.histplot(x=num_data[i], data=data, ax=axes[i], kde=True,
                 color = "darkseagreen")
plt.tight_layout()
plt.show()

# for i in num_data:   
#     fig = plt.figure()
#     sns.histplot(data[i],kde=True, color="cornflowerblue")
#     plt.tight_layout()
#     plt.show()


# Plot the correlation matrix 
cmap = sns.diverging_palette(200,20,sep=20,as_cmap=True)
corr_matrix = data_dummies.corr()
fig, ax = plt.subplots(figsize=(13, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5,
                 fmt=".2f", cmap=cmap)


# fit the models
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report,roc_auc_score,roc_curve 
from sklearn.metrics import plot_roc_curve,confusion_matrix

# Split data into train and test sets
np.random.seed(1)

X = data_dummies.drop("stroke_1", axis = 1)
y = data_dummies["stroke_1"]

from sklearn.model_selection import train_test_split,cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
len(y_train)


def plot_conf_mat(y_test_pred, y_pred):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    sns.heatmap(confusion_matrix(y_test, y_pred), #,normalize='true'
                annot=True,cbar=True, fmt='g',cmap=sns.color_palette('Blues'))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")


##### fitting the models without SMOTE
# logistic regression
from sklearn.linear_model import LogisticRegression
# define pipeline
steps = [("scaler", StandardScaler()),
         ('model', LogisticRegression())]
logReg_pip = Pipeline(steps=steps)
# evaluate pipeline
y_train_pred_lr = cross_val_predict(logReg_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_lr))
# Plot ROC curve and calculate and calculate AUC metric
logReg_pip.fit(X_train, y_train)
plot_roc_curve(logReg_pip, X_test, y_test)
y_test_pred_lr=logReg_pip.predict(X_test)
print(classification_report(y_test, y_test_pred_lr))

plot_conf_mat(y_test, y_test_pred_lr)

## KNN
from sklearn.neighbors import KNeighborsClassifier

steps = [("scaler", StandardScaler()),  
         ('model', KNeighborsClassifier())]
KNN_pip = Pipeline(steps=steps)

y_train_pred_knn = cross_val_predict(KNN_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_knn))
KNN_pip.fit(X_train, y_train)
plot_roc_curve(KNN_pip, X_test, y_test)
y_test_pred_knn=KNN_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_knn)
print(classification_report(y_test, y_test_pred_knn))

## Random Forest
from sklearn.ensemble import RandomForestClassifier

steps = [("scaler", StandardScaler()),
         ('rfc', RandomForestClassifier(random_state=1))]
RF_pip = Pipeline(steps=steps)
    
y_train_pred_rf = cross_val_predict(RF_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_rf))
RF_pip.fit(X_train, y_train)
plot_roc_curve(RF_pip, X_test, y_test)
y_test_pred_rf=RF_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_rf)
print(classification_report(y_test, y_test_pred_rf))

# tuning the hyperparameters
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

params={
    'rfc__max_depth': [10, 20],
    'rfc__n_estimators': [50, 100, 200, 400],
    'rfc__max_features': ['auto', 'sqrt'],
    'rfc__min_samples_leaf': [2,4,8],
    'rfc__min_samples_split': [10,20]   
}

rf_grid = GridSearchCV(RF_pip, params, cv=3,n_jobs=-1,scoring="f1")
rf_grid.fit(X_train, y_train)
print("Best Parameters for Model:  ",rf_grid.best_params_)

## Best Parameters for Model: 
##  {'rfc__max_depth': 10, 'rfc__max_features': 'auto', 
##   'rfc__min_samples_leaf': 2, 'rfc__min_samples_split': 10, 
##   'rfc__n_estimators': 50}

steps = [("scaler", StandardScaler()),
         ('rfb', RandomForestClassifier(random_state=1, max_features= 'auto', 
                                        min_samples_leaf= 2, n_estimators= 50,
                                        min_samples_split= 10))]

RFB_pip = Pipeline(steps)
y_pred_rfb = cross_val_predict(RFB_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_pred_rfb))

RFB_pip.fit(X_train, y_train)
plot_roc_curve(RFB_pip, X_test, y_test)
y_test_pred_rfb=RFB_pip.predict(X_test)
print(classification_report(y_test, y_test_pred_rfb))

plot_conf_mat(y_test, y_test_pred_rfb)

# plot the confusion matrix
confcmap = sns.light_palette("seagreen",as_cmap=True)
f, ax = plt.subplots(1,3,figsize=(15,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred_lr), ax=ax[0], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[0].set_title("Logistic Regression")
sns.heatmap(confusion_matrix(y_test, y_test_pred_knn), ax=ax[1], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[1].set_title("KNN")
sns.heatmap(confusion_matrix(y_test, y_test_pred_rfb), ax=ax[2], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[2].set_title("tuned Random Forest")
ax[0].set_xlabel("Predicted label")
ax[1].set_xlabel("Predicted label")
ax[2].set_xlabel("Predicted label")
ax[0].set_ylabel("True label")
plt.tight_layout()
plt.show()



## applying SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
over = SMOTE(sampling_strategy=0.1, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=1)

steps = [("scaler", StandardScaler()), ('over', over), ('under', under), 
         ('model', LogisticRegression())]
logRegS_pip = Pipeline(steps=steps)
# evaluate pipeline
y_train_pred_lrs = cross_val_predict(logRegS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_lrs))
# Plot ROC curve and calculate and calculate AUC metric
logRegS_pip.fit(X_train, y_train)
plot_roc_curve(logRegS_pip, X_test, y_test)
y_test_pred_lrs=logRegS_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_lrs)
print(classification_report(y_test, y_test_pred_lrs))


# borderline-SMOTE for imbalanced dataset
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

oversampleB = BorderlineSMOTE(random_state=1)
X_trainb, y_trainb = oversampleB.fit_resample(X_train, y_train)

oversampleS = SVMSMOTE(random_state=1)
oversampleA = ADASYN(random_state=1)

steps = [("scaler", StandardScaler()),('over', over), ('under', under), 
         ('model', LogisticRegression())]
logRegBS_pip = Pipeline(steps=steps)
# evaluate pipeline
y_train_pred_lrBS = cross_val_predict(logRegBS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_lrBS))
# Plot ROC curve and calculate and calculate AUC metric
logRegBS_pip.fit(X_train, y_train)
plot_roc_curve(logRegBS_pip, X_test, y_test)
y_test_pred_lrBS=logRegBS_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_lrBS)
print(classification_report(y_test, y_test_pred_lrBS))



## KNN
steps = [("scaler", StandardScaler()),('over', over), ('under', under),  
         ('model', KNeighborsClassifier())]
KNNS_pip = Pipeline(steps=steps)

y_train_pred_KNNS = cross_val_predict(KNNS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_KNNS))
KNNS_pip.fit(X_train, y_train)
plot_roc_curve(KNNS_pip, X_test, y_test)
y_test_pred_KNNS=KNNS_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_KNNS)
print(classification_report(y_test, y_test_pred_KNNS))



## Random Forest
steps = [("scaler", StandardScaler()),
         ('over', over), ('under', under),  
         ('rfc', RandomForestClassifier(random_state=1))]
RFS_pip = Pipeline(steps=steps)
    
y_train_pred_rfS = cross_val_predict(RFS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_train_pred_rfS))
RFS_pip.fit(X_train, y_train)
plot_roc_curve(RFS_pip, X_test, y_test)
y_test_pred_rfS=RFS_pip.predict(X_test)
plot_conf_mat(y_test, y_test_pred_rfS)
print(classification_report(y_test, y_test_pred_rfS))


params={
    'rfc__max_depth': [10, 20],
    'rfc__n_estimators': [50, 100, 200, 400],
    'rfc__max_features': ['auto', 'sqrt'],
    'rfc__min_samples_leaf': [2,4,8],
    'rfc__min_samples_split': [10,20]   
}

# ## Grid search
# RFS_grid = RandomizedSearchCV(RFS_pip, params, cv=3,n_jobs=-1,scoring="f1")
# RFS_grid.fit(X_train, y_train)
# print("Best Parameters for Model:  ",RF_grid.best_params_)

RFS_Random = RandomizedSearchCV(RFS_pip, params, cv=3,n_jobs=-1,
                                scoring="f1", random_state=1)
RFS_Random.fit(X_train, y_train)
print("Best Parameters for Model:  ",RFS_Random.best_params_)

# Best Parameters for Model:   {'rfc__n_estimators': 100,
#                               'rfc__min_samples_split': 10, 
#                               'rfc__min_samples_leaf': 8, 
#                               'rfc__max_features': 'auto', 
#                               'rfc__max_depth': 20}

steps = [("scaler", StandardScaler()),
         ('over', over), ('under', under),  
         ('rfb', RandomForestClassifier(random_state=1, max_features= 'auto', 
                                        min_samples_leaf= 8, n_estimators= 100,
                                        min_samples_split=10, max_depth=20))]
RFBS_pip = Pipeline(steps)

y_pred_RFBS = cross_val_predict(RFBS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_pred_RFBS))
RFBS_pip.fit(X_train, y_train)
plot_roc_curve(RFBS_pip, X_test, y_test)

y_test_pred_RFBS=RFBS_pip.predict(X_test)
print(classification_report(y_test, y_test_pred_RFBS))

plot_conf_mat(y_test, y_test_pred_RFBS)

## importance of variables
importances = RFBS_pip.steps[3][1].feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]-1):
    print("%2d) %-*s %f" % (f, 30, X_train.columns[indices[f]], importances[indices[f]]))


for i in range(X_train.shape[1]-1):
    plt.barh(i,importances[indices[i]],align='center', 
             color=sns.cubehelix_palette(len(indices), start = 1, rot = -.75)[i])
    plt.yticks(np.arange(X_train.shape[1]),X_train.columns, fontsize=12)
plt.show()


## Grid search
RFS_grid = GridSearchCV(RFS_pip, params, cv=3,n_jobs=-1, scoring="f1")
RFS_grid.fit(X_train, y_train)
print("Best Parameters for Model:  ",RFS_grid.best_params_)

steps = [("scaler", StandardScaler()),
         ('over', over), ('under', under),  
         ('rfb', RandomForestClassifier(random_state=1, max_features= 'auto', 
                                        min_samples_leaf= 2, n_estimators= 400,
                                        min_samples_split= 10, max_depth=10))]
                                        
RFBS_pip = Pipeline(steps)

y_pred_RFBS = cross_val_predict(RFBS_pip, X_train, y_train, cv = 10)
print(classification_report(y_train, y_pred_RFBS))
RFBS_pip.fit(X_train, y_train)
plot_roc_curve(RFBS_pip, X_test, y_test, color="seagreen")

y_test_pred_RFBS=RFBS_pip.predict(X_test)
print(classification_report(y_test, y_test_pred_RFBS))

plot_conf_mat(y_test, y_test_pred_RFBS)




# plot the ROC curve
confcmap = sns.light_palette("seagreen",as_cmap=True)
f, ax = plt.subplots(1,3,figsize=(15,4))
plot_roc_curve(logRegS_pip, X_test, y_test, ax=ax[0], color="seagreen")
ax[0].set_title("Logistic Regression")
plot_roc_curve(KNNS_pip, X_test, y_test, ax=ax[1], color="seagreen")
ax[1].set_title("KNN")
plot_roc_curve(RFBS_pip, X_test, y_test, ax=ax[2], color="seagreen")
ax[2].set_title("tuned Random Forest")
plt.tight_layout()
plt.show()



# plot the confusion matrix
confcmap = sns.light_palette("seagreen",as_cmap=True)
f, ax = plt.subplots(1,3,figsize=(15,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred_lrs), ax=ax[0], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[0].set_title("Logistic Regression")
sns.heatmap(confusion_matrix(y_test, y_test_pred_KNNS), ax=ax[1], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[1].set_title("KNN")
sns.heatmap(confusion_matrix(y_test, y_test_pred_RFBS), ax=ax[2], 
            annot=True,cbar=True, fmt='g',cmap=confcmap)
ax[2].set_title("tuned Random Forest")
ax[0].set_xlabel("Predicted label")
ax[1].set_xlabel("Predicted label")
ax[2].set_xlabel("Predicted label")
ax[0].set_ylabel("True label")
plt.tight_layout()
plt.show()








