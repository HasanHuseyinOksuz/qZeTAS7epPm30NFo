# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:13:26 2020

@author: HASAN HÜSEYİN ÖKSÜZ
"""

import pandas as pd 
from sklearn.model_selection import train_test_split  

from sklearn import tree
#import graphviz # decision tree visualization 

import statsmodels.api as sm # statistical library # BONUS

columns_x = ["X1", "X2","X3", "X4", "X5", "X6"] #features
columns_y = ["Y"] #target

# read data from csv
data = pd.read_csv("ACME-HappinessSurvey2020.csv") 


df_x = pd.DataFrame(data, columns = columns_x) # inputs
df_y = pd.DataFrame(data, columns = columns_y) # output

#split data into train = 0.85, test = 0.15
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.15, random_state=126)


#print(x_train.shape, x_test.shape)
#print(x_test)

#---------------------------- Decision Tree -----------------------------------


#create model
clf = tree.DecisionTreeClassifier()

# train the model
clf.fit(x_train , y_train)

# uncommnet if want to see as pdf in ./ folder  
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data)
#graph.render("Decision Tree") 

# evaluate the model 
score = clf.score(x_test, y_test)
print(score)

# predict new data
#predict = clf.predict([[5, 3, 3, 3, 3, 5]]) 
#print(predict)


#------------------------------------ BONUS -----------------------------------
#Logit Regression
sm_model = sm.Logit(df_y, sm.add_constant(df_x)).fit(disp=0)

#summary of regression
print(sm_model.summary())

#if p values smaller than 0.05, thats mean is statistically significant.
# Thus, we can remove all of features but X1.  
print(sm_model.pvalues)

  
























