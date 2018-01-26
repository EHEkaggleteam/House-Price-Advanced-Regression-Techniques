# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:18:02 2018

@author: frank
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

estate=pd.read_csv('C:\\Users\\frank\\Desktop\\PYTHON\\Kaggle\\House\\train_manipulated.csv')
estate=estate.dropna(axis=1,thresh=700)#axis=1表示丢弃列，thresh=700表示丢弃非空值小于700的列
#type(estate)
print(estate.shape)

x=estate.iloc[:,1:76]#丢弃空值较多的列后还剩77列，其中第一列是ID不要
#type(x)
#print(x[:5,:])
y=estate[['SalePrice']]
#type(y)

from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)
x=vec.fit_transform(x.to_dict(orient='record'))
x.shape#为什么没有将连续变量也encoder了

#ss_x=StandardScaler()#数据里有空值，无法standardize
#x=ss_x.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

clfX=xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10)

clfX.fit(x_train,y_train)

print('\nXGB_train_score=%f\n'%clfX.score(x_train,y_train))
print('\nXGB_test_score=%f\n'%clfX.score(x_test,y_test))

unknown=pd.read_csv('C:\\Users\\frank\\Desktop\\PYTHON\\Kaggle\\House\\test_manipulated.csv')
unknown=unknown.dropna(axis=1,thresh=700)
x_unknown=unknown.iloc[:,1:76]
#print(x_unknown.head())
x_unknown=vec.transform(x_unknown.to_dict(orient='record'))
#x_unknown=ss_x.transform(x_unknown)
x_unknown.shape
y_unknown=clfX.predict(x_unknown)
#type(Xy_unknown)
#print(y_unknown.shape)
#由于y_unknown是一个ndarray，必须转化成pd.dataframe才能使用pd.to_excel

df=pd.DataFrame(y_unknown,index=np.arange(1460,2919,step=1),columns=['SalePrice'])#生成一个DataFrame，df
df.to_csv('C:\\Users\\frank\\Desktop\\PYTHON\\Kaggle\\House\\MySubmission_XGB.csv')
