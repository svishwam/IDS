import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt


#===================================INPUT======================================
#import dataset
Dataset=pd.read_csv('Kdd_dataset.csv')

print (Dataset.head(10))
print(Dataset.info())
print(Dataset.describe())

#================================PRE_PROCESSING================================
#Data Preprocessing
#check missing values
print ('Dataset contain null:\t',Dataset.isnull().values.any())
print ('Describe null:\n',Dataset.isnull().sum())
print ('No of  null:\t',Dataset.isnull().sum().sum())


#Selecting Independent variable 
x=Dataset.drop("class",axis = 1).values
x1=pd.DataFrame(x)

#Selecting Dependent variable
y=Dataset['class'].values
k1=pd.DataFrame(y)

#class lable converting
for i in range(9999):
    if y[i]=='normal':
        y[i]=0
    else:
        y[i]=1
type(y)
#type(x)
y=y.astype('int')


#==============================LABEL ENCOADING=================================
#Encoding categorical data
from sklearn.compose import ColumnTransformer 
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
Y=pd.DataFrame(x[:,1])

labelencoder_x=LabelEncoder()
x[:,2]=labelencoder_x.fit_transform(x[:,2])
Y=pd.DataFrame(x[:,2])

labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
Y=pd.DataFrame(x[:,3])

ct = ColumnTransformer([("protocol_type", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)
labelencoder_x=LabelEncoder()
ct = ColumnTransformer([("service", OneHotEncoder(), [2])], remainder = 'passthrough')
x = ct.fit_transform(x)
ct = ColumnTransformer([("flag", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)

#=============================MODEL SELECTION==================================
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print(X_train)
print(x)

#===========================Dimensionality Reduction===========================
'''Principle Component Analysis'''

pca_dims = PCA()
pca_dims.fit(X_train)

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)
X_test_reduced = pca.transform(X_test)
print("\nreduced shape: " + str(X_reduced.shape))
print("\nrecovered shape: " + str(X_recovered.shape))


#====================================RANDOM FOREST=======================================
'''Random Forest'''

print('\nRandom Forest')
rf = RandomForestClassifier()
rf.fit(X_reduced,y_train)

rf_pred = rf.predict(X_test_reduced)
RF=accuracy_score(y_test, rf_pred)*100
print("RF accuracy is: ",RF,'%')
print()
print('Classification Report')
rf_cr=classification_report(y_test, rf_pred)
print(rf_cr)


skplt.estimators.plot_learning_curve(RandomForestClassifier() , X_reduced, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="RF Digits Classification Learning Curve");
plt.show()
