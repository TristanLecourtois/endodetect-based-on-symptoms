# endodetect-based-on-symptoms
This repository focuses on using machine learning (ML) to develop a self-diagnostic endometriosis tool based solely on patient-reported symptoms

## Table of Contents
1. [Introduction](#introduction)
2. [Machine Learning Algorithms](#machine-learning-algorithms)
   - [Logistic Regression](#logistic-regression)
   - [Decision Trees](#decision-trees)
   - [Random Forest](#random-forest)
3. [Dataset](#dataset)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [License](#license)

## Introduction
Current endometriosis diagnostic methods are invasive and expensive, prompting the need for non-invasive screening tools. Various studies have explored biomarkers, genomic data, and patient-reported symptoms, but none have been entirely successful in replacing laparoscopy. This project presented focuses on using machine learning
(ML) to develop a self-diagnostic tool based solely on patient-reported symptoms. The goal is to create an easy-to-use model for women in the early stages of medical investigation, providing a preliminary indication of their likelihood of having endometriosis. The study identifies a set of 24 symptoms most effective for endometriosis prediction, achieving high sensitivity (0.93) and specificity (0.95) on holdout data. The aim is to reduce the time-to-diagnosis and provide insights into the importance of different symptoms in predicting endometriosis.

## Machine Learning Algorithms
We applied several ML algorithms to train multiple endometriosis prediction models. Specifically, we applied decision trees, Random Forest and Logistic Regression. Besides generating predictions, these models also provide an importance analysis feature, which can be used to identify and remove non-contributing features from future surveys. Model performance was evaluated using common ML metrics : ac-
curacy, sensitivity (recall), specificity, precision, F1-score,area under the ROC curve (AUC) and Matthew Coorelation Coefficient. To ensure significance of the results, we used a ten-fold cross-validation procedure.

### Logistic Regression

Description and usage of Logistic Regression.

### Decision Trees

Description and usage of Decision Trees.

### Random Forest

Description and usage of Random Forest

## Dataset 



## Results


## Installation 

These instructions assume you have `git` installed for working with Github from command window.

1. Clone the repository, and navigate to the downloaded folder. Follow below commands.

```
git clone https://github.com/TristanLecourtois/endodetect-based-on-symptoms.git
cd endodetected-based-on-symptoms
```
2. Install few required pip packages, which are specified in the requirements.txt file.

```
pip3 install -r requirements.txt
```

## License 
The code in this project is licensed under the MIT license 2024 - Tristan Lecourtois.


