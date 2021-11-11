import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
df_train  = pd.read_csv('housetrain.csv')
X = df_train.iloc[:, :-1].values
y = df_train.iloc[:, -1].values

df_train['SalePrice'].describe() # afficher count mean  std  minimum etc .....
sns.distplot(df_train['SalePrice'])#afficher histogramme de sale price 

print("Skewness: %f" % df_train['SalePrice'].skew()) #monter la position de l'asymetrie 
print("Kurotsis: %f" % df_train['SalePrice'].kurt()) # twarri 9adech tfarti7 
#ascatter plot grilivearea/saleprice

var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)# data fehe sale price oul var elli hia griliv area 
data.plot.scatter(x=var,y='SalePrice',ylim=(0.800000));#tamel visualisation bi ni9at (scatter)

#scatter plot totalbsmtsf/saleprice
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1) # nefs le7keye 
data.plot.scatter(x=var,y='SalePrice',ylim=(0.800000))

#scatter plot OverallQual/saleprice
var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #  el courbe mch wadha7 bel behi donc namloulou faza okhra mta plot 
data.plot.scatter(x=var,y='SalePrice',ylim=(0.800000))

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(10, 8)) #afficher la courbe de f en fonction de x avec adn fig size el kobr 
fig = sns.boxplot(x=var, y="SalePrice", data=data)# twarrina tsawaer en box thot fehe x ou y ou data mtaek 
fig.axis(ymin=0, ymax=800000);# les valeur de y 
#box plot overallqual/saleprice
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);#roatate les  labels des valeurs de l'axe x 

#correlation matrix
corrmat = df_train.corr() # effectuer le comatrice de correlation 
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True); # vlmax :Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
   
                                         #If True, set the Axes aspect to “equal” so each cell will be square-shaped.
#saleprice correlation matrix
k = 10 #number of variables for heatmap # takhou 10 variable mouhemmine 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index #   takhou les key mta valeurs :Return the first n rows with the largest values in columns 
cm = np.corrcoef(df_train[cols].values.T)# tati les valeur mta correlation mta kol contenue 
sns.set(font_scale=1.25) # afficher le 5at 
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'] #♣ taffichi kolchey 
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) # tati el  el missing data be tertib 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)













































# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)