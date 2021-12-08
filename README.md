# Stroke-Prediction

## Introduction

Stroke is the 2nd leading cause of death globally according to the World Health Organization (WHO), responsible for approximately 11\% of total deaths. A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications.

In this report, we try to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Our top priority in this health problem is to identify patients with a stroke.


## Data exploration


The dataset is downloaded from [stroke prediction dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset), with 5110 rows and 12 columns. Each row in the data provides relavant information about the patient including:

* id: unique identifier.
* gender: "Male", "Female" or "Other".
* age: age of the patient.
* hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension.
* heart\_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease.
* ever\_married: "No" or "Yes".
* work\_type: "children", "Govt\_jov", "Never\_worked", "Private" or "Self-employed".
* Residence\_type: "Rural" or "Urban".
* avg\_glucose\_level: average glucose level in blood.
* bmi: body mass index.
* smoking\_status: "formerly smoked", "never smoked", "smokes" or "Unknown" where "Unknown" means that the information is unavailable for this patient.
* stroke: 1 if the patient had a stroke or 0 if not.


There are 3.93\% missing data of BMI variable and we impute them with the mean of BMI.  The response variable "Stroke" from the bottom right is highly imbalanced, with 4861 subjects without any stroke while only 249 subjects with a stroke. There is one subject with gender as other and will be deleted for further convenience.

## Model selection

We fit logistic regression as a benchmark model and compare it with KNN classifier and Random Forest. 30\% of the data is randomly chosen as the test set and models are fit based on 70\% of the data left.

Before applying SMOTE, we fit logistic regression, KNN and random forest tuned based on grid search with cross validation. Even though all three models get a high accuracy as around 0.95, they tend to predict all the subjects in the test set to be free of stroke, since the respondent variable stroke is so imbalanced that the non-stroke subjects would not make a huge difference on the outcome.


To deal with this issue, we introduce Synthetic Minority Oversampling Technique, or SMOTE for short, which is just oversampling the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model. SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.

Since the original paper on SMOTE(Nitesh Chawla, et al. 2002) suggested combining SMOTE with random undersampling of the majority class. So for our model, we first oversample the minority class to have 10 percent the number of examples of the majority class, then use random undersampling to reduce the number of examples in the majority class to have twice the number of minority class.

After introducing SMOTE to the training set, we fit logistic regression, KNN  and random forest classifier. Randomized search with cross validation is applied and the best parameters are found. As we can observe from the confusion matrices, all three models do better on the test set with the minority class of the variable stroke. 54 out of 83 subjects with strokes are correctly classified by logistic regression, which is the best result of the three models with the highest recall of the minority class 1 as 0.65. However, logistic regression compromises its accuracy on the majority class a lot for better prediction on the minority class. By considering the F1-score, Random Forest is the best model for over-all performance.

Feature importances are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree in python. We observe that, as expected, the three first features are found important. We observe that the age of the subject is the most important feature with its importance as about 0.48. The average glucose level and bmi of the subjects are also found important when predicting stroke.


To improve the performance of our models, we can try other proportions of oversampling and undersampling and choose the best one based on cross validation. However, it may result in over fitting and weaken the model's ability to generalize. We can also try different techniques of balancing the data such as Borderline SMOTE, SVM SMOTE, ADASYN, etc.
