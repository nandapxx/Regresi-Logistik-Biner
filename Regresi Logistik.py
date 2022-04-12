# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:50:07 2022

@author: ananda putri
"""

import pandas as pd
df = pd.read_csv (r'C:\Users\ananda putri\Documents\R script\Regresi Logistik.csv',
                  header=0, sep=';')
df = pd.DataFrame(df)
df = df.iloc[:,1:14]
df.iloc[0,:]
print(df.columns)
print (df.head())
df.info()

"Menyamakan jenis data tiap kolom sebagai numerik"
df = df.apply (pd.to_numeric, errors='coerce')
df = df.dropna()
df.info()
df.describe()

"Mendeteksi dan menghapus outlier"
"Melihat boxplot"
import matplotlib.pyplot as plt
df.boxplot()
def plot_boxplot(df,cl):
    df.boxplot(column=[cl])
    plt.grid(False)
    plt.show()
plot_boxplot(df,'NUM_CHILDREN')
plot_boxplot(df,'INCOME')

"Upper limit dan Lower limit tiap Variabel dengan IQR"
low = .25
high = .75
quant_df = df.quantile([low, high])
print(quant_df)

Q1= df.quantile(0.25)
Q3= df.quantile(0.75)
IQR= Q3-Q1
UL= Q3+(IQR*1.5)
LL= Q1-(IQR*1.5)
UL['APPROVED_CREDIT']

"Membuat fungsi outlier"
def outliers (df,cl):
    Q1 = df[cl].quantile(0.25)
    Q3 = df[cl].quantile(0.75)
    IQR = Q3-Q1
    
    UL = Q3+(IQR*1.5)
    LL = Q1-(IQR*1.5)
    
    ls = df.index[(df[cl] < LL) | (df[cl]) > UL]
    return ls

index_list = []
for var in ['TARGET', 'GENDER','NUM_CHILDREN','INCOME',
            'ANNUITY','PRICE','DAYS_AGE','DAYS_WORK',
            'DAYS_REGISTRATION','DAYS_ID_CHANGE',
            'HOUR_APPLY','APPROVED_CREDIT','STATUS']:
    index_list.extend(outliers(df,var))
index_list[0:5]

"Menghapus outlier"
def remove(df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df
df1 = remove(df,index_list)
df1.shape
df.shape

"Melihat Korelasi tiap variabel terhadap status"
df.corr(method ='pearson')
df['TARGET'].corr(df['STATUS'])
df['GENDER'].corr(df['STATUS'])
df['NUM_CHILDREN'].corr(df['STATUS'])
df['INCOME'].corr(df['STATUS'])
df['ANNUITY'].corr(df['STATUS'])
df['PRICE'].corr(df['STATUS'])
df['DAYS_AGE'].corr(df['STATUS'])
df['DAYS_WORK'].corr(df['STATUS'])
df['DAYS_REGISTRATION'].corr(df['STATUS'])
df['DAYS_ID_CHANGE'].corr(df['STATUS'])
df['HOUR_APPLY'].corr(df['STATUS'])
df['APPROVED_CREDIT'].corr(df['STATUS'])
df['STATUS'].corr(df['STATUS'])

"Model regresi logistik terbaik"
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn

X = df[['TARGET', 'GENDER','NUM_CHILDREN','INCOME',
        'ANNUITY','PRICE','DAYS_AGE','DAYS_WORK','DAYS_REGISTRATION',
        'DAYS_ID_CHANGE','HOUR_APPLY', 'APPROVED_CREDIT']]
y = df['STATUS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
log_reg= LogisticRegression()
log_reg.fit(X_train,y_train)
log_odds = log_reg.coef_[0]
log_reg.intercept_
pd.DataFrame(log_odds, 
             X.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
y_pred=log_reg.predict(X_test)

"Melihat variabel terpenting"
for i,v in enumerate(log_odds):
	print('Feature: %0d, Score: %.7f' % (i,v))
plt.bar([x for x in range(len(log_odds))], log_odds)
plt.show()


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))




