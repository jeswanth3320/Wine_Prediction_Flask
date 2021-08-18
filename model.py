import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('winequalityN.csv')
df['fixed acidity'].fillna((df['fixed acidity'].mean()), inplace=True)
df['volatile acidity'].fillna((df['volatile acidity'].mean()), inplace=True)
df['citric acid'].fillna((df['citric acid'].mean()), inplace=True)
df['residual sugar'].fillna((df['residual sugar'].mean()), inplace=True)
df['pH'].fillna((df['pH'].mean()), inplace=True)
df['sulphates'].fillna((df['sulphates'].mean()), inplace=True)
df['chlorides'].fillna((df['chlorides'].mean()), inplace=True)
new_df=df.drop('total sulfur dioxide',axis=1)
new_df = pd.get_dummies(new_df,drop_first=True)
new_df['best quality']=[ 1 if x>=7 else 0 for x in df.quality]
new_df['best quality']
x=new_df.drop(['quality','residual sugar','free sulfur dioxide'],axis=1).values
x1=x=new_df.drop(['quality','residual sugar','free sulfur dioxide'],axis=1)
y=new_df['quality'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)


from sklearn.preprocessing import MinMaxScaler
# creating normalization object
norm = MinMaxScaler()
# fit data
norm_fit = norm.fit(x_train)
new_xtrain = norm_fit.transform(x_train)
new_xtest = norm_fit.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

rnd = RandomForestClassifier()
# fit data
fit_rnd = rnd.fit(new_xtrain,y_train)
# predicting score

rnd_score = rnd.score(new_xtest,y_test)
pickle.dump(rnd,open('data.pkl','wb'))

lr = pickle.load(open('data.pkl', 'rb'))

ans=rnd.predict([[7.0,0.270,0.36,0.045,1.00100,3.00,0.450000,8.8,1,0]])[0]
