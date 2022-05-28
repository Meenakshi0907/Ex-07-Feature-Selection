# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df=pd.read_csv("titanic_dataset.csv")

df

df.isnull().sum()

df["Embarked"]=df["Embarked"].fillna(df['Embarked'].mode()[0])
df["Age"]=df["Age"].fillna(df["Age"].median())

df.isnull().sum()

df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

df.boxplot()

cols = ["Age","SibSp","Parch","Fare"]
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

df.boxplot()

from category_encoders import BinaryEncoder
be=BinaryEncoder()
newdata=be.fit_transform(df[["Sex"]])
df["Sex"]=newdata
df

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
temp=["C","Q","S"]
oe=OrdinalEncoder(categories=[temp])
df["Embarked"]=oe.fit_transform(df[["Embarked"]])
df

from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
df2=pd.DataFrame(rs.fit_transform(df),columns=['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

df2.skew()

df_skew=pd.DataFrame()

df_skew["Survived"]=np.sqrt(df2["Survived"])
df_skew["Pclass"]=qt.fit_transform(df[["Pclass"]])
df_skew["Sex"],parameters=stats.yeojohnson(df["Sex"])
df_skew["Age"]=np.sqrt(df["Age"])
df_skew["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df_skew["Fare"],parameters=stats.yeojohnson(df["Fare"])
df_skew["Embarked"],parameters=stats.yeojohnson(df["Embarked"])

df_skew.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df_skew.drop("Survived",1) 
y = df_skew["Survived"]          

plt.figure(figsize=(12,10))
cor = df_skew.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Set3)
plt.show()

cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

#Wrapper method
X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
model.pvalues

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 4)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, 2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# Embedded Method
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
# OUPUT
![output](./fs1.png)
![output](./fs2.png)
![output](./fs3.png)
![output](./fs4.png)
![output](./fs5.png)
![output](./fs6.png)
![output](./fs7.png)
![output](./fs8.png)
![output](./fs9.png)
![output](./fs10.png)
![output](./fs11.png)
![output](./fs12.png)

# Result:
Thus, the various feature selection techniques have been performed on a given dataset successfully.