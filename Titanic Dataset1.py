#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import set_config
from imblearn.pipeline import Pipeline as imbPipeline ,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler , OrdinalEncoder
from feature_engine.selection import DropFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import make_scorer , precision_score , recall_score , accuracy_score
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_predict
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading data and exploring 

# In[2]:


Titanic_dataset=pd.read_csv(r"C:\Users\Administrator\Desktop\my_lab\Titanic\train.csv")


# In[3]:


Titanic_dataset.head()


# In[4]:


Titanic_dataset.info()


# In[5]:


Titanic_dataset.describe().T


# In[6]:


Titanic_dataset.isnull().sum()


# In[7]:


# What percentage survived and unsurvived
percentage = Titanic_dataset["Survived"].value_counts(normalize=True) * 100
percentage


# In[8]:


numerical_var=["Age" , 'Fare']
for col in numerical_var:
    plt.hist(data=Titanic_dataset , x=col, bins=20)
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.show()


# In[12]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
sns.countplot(x="Sex" , hue='Survived' , data=Titanic_dataset, ax=axes[0])
axes[0].set_title('Bar Plot Sex')
sns.countplot(x="Pclass" , hue='Survived' , data=Titanic_dataset, ax=axes[1])
axes[1].set_title('Bar Plot Pclass')
sns.countplot(x="Embarked" , hue='Survived' , data=Titanic_dataset , ax=axes[2])
axes[2].set_title('Bar Plot Embarked')
plt.show()


# In[13]:


Titanic_dataset.corr().round(2)


# In[14]:


sns.heatmap(Titanic_dataset.corr())


# ## Hypothesis

# In[15]:


survival_rates = Titanic_dataset.groupby('Pclass')['Survived'].mean()
chi2, p_value, _, _ = stats.chi2_contingency(Titanic_dataset.groupby(['Pclass', 'Survived']).size().unstack())
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in survival rates between passenger classes.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in survival rates between passenger classes.")
print(survival_rates)


# In[16]:


survival_rates = Titanic_dataset.groupby('Embarked')['Survived'].mean()
chi2, p_value, _, _ = stats.chi2_contingency(Titanic_dataset.groupby(['Embarked', 'Survived']).size().unstack())
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in survival rates between Embarked classes.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in survival rates between Embarked.")
print(survival_rates)


# In[17]:


survival_rates = Titanic_dataset.groupby('Sex')['Survived'].mean()
chi2, p_value, _, _ = stats.chi2_contingency(Titanic_dataset.groupby(['Sex', 'Survived']).size().unstack())
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in survival rates between Sex classes.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in survival rates between Sex.")
print(survival_rates)


# ## DataPreprocessing

# ## Adding New Feature

# In[18]:


class FamilySizeFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        dataset = X.copy()
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        bins = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]
        labels = ['Alone', 'Small', 'Small', 'Small', 'Small', 'Small', 'Large', 'Large', 'Large']
        dataset['FamilySizeCategory'] = pd.cut(dataset['FamilySize'], bins=bins, labels=labels, right=False, ordered=False)
        return dataset.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)


# In[19]:


Titanic_dataset['Fare_log'] = np.log1p(Titanic_dataset["Fare"])


# # Training

# ## split_train_test

# In[20]:


y=Titanic_dataset["Survived"]
X=Titanic_dataset.drop('Survived' , axis=1)


# In[21]:


X_train, X_test , y_train , y_test=train_test_split(X , y , test_size=0.2 , random_state=42, stratify = y)


# In[22]:


X_train.head()


# ## Pipeline & Model Selection

# In[23]:


list_drop=list(X_train[['PassengerId' , 'Name' , 'Cabin' , 'Ticket' , 'Fare']])
list_cat=list(X_train[["Sex" , "Embarked" , 'Pclass']])
list_num=list(X_train[["Age" , "Fare_log" ]])


# In[24]:


# Linear model 
lr = LogisticRegression(warm_start=True, max_iter=400)
# RandomForest
rf = RandomForestClassifier()
# XGB
xgb = XGBClassifier(tree_method="hist", verbosity=0, silent=True)
# Ensemble
lr_xgb_rf = VotingClassifier(estimators=[('lr', lr), ('xgb', xgb), ('rf', rf)], voting='soft')


# In[25]:


ppl=imbPipeline([
    ('familysize' , FamilySizeFeature() ),
    ('drop_col' , DropFeatures(list_drop)),
    ('cleaning' , ColumnTransformer([
  
        
        ('number' , make_pipeline(SimpleImputer(strategy="mean") , MinMaxScaler()) ,list_num ), 
        ('category' , make_pipeline(SimpleImputer(strategy='most_frequent') ,OneHotEncoder(handle_unknown='ignore')), list_cat), 
        ('category0' , make_pipeline(OrdinalEncoder()), ["FamilySizeCategory"]), 
         ])),
    ('smote' , SMOTE()),
    ('ensemble' , lr_xgb_rf),
    
    ])


# In[26]:


set_config(display="diagram")
ppl.fit(X_train, y_train)


# In[28]:


y_pred = cross_val_predict(ppl , X_train, y_train , cv=3) 
print('Accuracy' , round ( accuracy_score(y_train, y_pred) , 2))
print('Precision:',round( precision_score(y_train, y_pred) , 2))
print('Recall:', round (recall_score(y_train, y_pred) , 2))


# In[ ]:





# In[ ]:




