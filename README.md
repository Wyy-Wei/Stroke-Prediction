# Stroke-Prediction

## Introduction

Stroke is the 2nd leading cause of death globally according to the World Health Organization (WHO), responsible for approximately 11\% of total deaths. A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications.

In this report, we try to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Our top priority in this health problem is to identify patients with a stroke.


## Data exploration


The dataset is downloaded from \href{stroke prediction dataset}{https://www.kaggle.com/fedesoriano/stroke-prediction-dataset}, with 5110 rows and 12 columns. Each row in the data provides relavant information about the patient including:

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


There are 3.93\% missing data of BMI variable and we impute them with the mean of BMI. Then we plot the histograms of the categorical variables as shown in Figure \ref{cat}. The response variable "Stroke" from the bottom right is highly imbalanced, with 4861 subjects without any stroke while only 249 subjects with a stroke. There is one subject with gender as other and will be deleted for further convenience.
