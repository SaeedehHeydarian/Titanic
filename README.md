# Titanic-Machine-Learning-from-Distaster
### Overview
The Titanic dataset is a famous dataset that contains information about the passengers onboard the RMS Titanic, which famously sank on its maiden voyage on April 15, 1912. The dataset is often used in data analysis and machine learning projects as a beginner's dataset to practice data exploration, data cleaning, and predictive modeling.
### Context
The dataset provides various attributes for each passenger, such as their age, sex, ticket class, cabin, fare, and whether they survived or not. The main goal is usually to predict whether a passenger survived based on the given features.

### Content
The dataset consists of the following columns:
PassengerId: Unique identifier for each passenger
Survived: Whether the passenger survived (0 = No, 1 = Yes)
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name: Passenger's name
Sex: Passenger's gender (male or female)
Age: Passenger's age in years
SibSp: Number of siblings/spouses aboard the Titanic
Parch: Number of parents/children aboard the Titanic
Ticket: Ticket number
Fare: Passenger fare
Cabin: Cabin number
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
### Usage 
The Titanic dataset is often used to explore data visualization, data preprocessing, and machine learning techniques. It has been used in competitions on platforms like Kaggle, where participants build predictive models to determine passenger survival.

#### Dataset source:  [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)




# Importing Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import (train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_validate)
from sklearn import set_config
from imblearn.pipeline import Pipeline as imbPipeline ,make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler , OrdinalEncoder
from feature_engine.selection import DropFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import make_scorer , precision_score , recall_score , accuracy_score
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_predict
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading Data and Exploring 

Titanic_dataset=pd.read_csv(r"train (1).csv")


