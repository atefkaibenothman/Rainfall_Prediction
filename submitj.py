# ====================== Preparation  ===================
# import library, module and training/evaluation data
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV

# Note: file is comma-delimited
X = np.genfromtxt("Data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("Data/kaggle.Y.train.txt",delimiter=',')
Xtest = np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')
X_dev = X[:20000,:]
Y_dev = Y[:20000]


# ====================== Stack implementation description ===================
# stack:
# 3 learners are used at level 0, namely extraTree, randomForest, gradientBoosting regressors
# 1 regressor (ridge) at level 1 is used to process outputs from learners at level 0
# n_trees is the number of estimators used for each of the three learners at level 0


# ====================== Training level-0 learners  ===================
n_trees = 50
# level 0 learners:
clfs = [
        ExtraTreesRegressor(n_estimators = n_trees *2),
        RandomForestRegressor(n_estimators = n_trees),
        GradientBoostingRegressor(n_estimators = n_trees)
    ]

# split data into (X1, Y1) and (X2, Y2)
# (X1, Y1) are used to train the 3 level-0 learners
# X2 is used to geneate temp_train from the three level-0 learners
# (temp_train, Y2) are used to train the level-1 learner
X1,X2,Y1,Y2 = ml.splitData(X,Y,0.75)


# temp_train are the intermediate training data, ie, outputs of the 3 level-0 learners, also inputs of the level-1 learner
temp_train = np.zeros((  len(Y2)   ,len(clfs)    ))
temp_test=np.zeros((  Xtest.shape[0]   ,len(clfs)    ))
for i, clf in enumerate(clfs):
    clf.fit(X1,Y1)                             # train each level-0 learner
    temp_train[:,i] = clf.predict(X2)          # intermediate data for level-1 learner given data X2 are generated
    temp_test[:,i] = clf.predict(Xtest)        # intermediate data for level-1 learner given data Xtest are also generated
    

# ====================== Training the level-1 learner  ===================
# level-1 learner
# cv = 5: 5 folds cross validation    
alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
stk = RidgeCV(alphas=alphas, normalize=True, cv=5).fit(temp_train, Y2)



# ====================== Prediction  ===================
# predict the test data and write output Y_hat to .csv file 
Y_hat = stk.predict(temp_test)
fh = open('n_50_predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(Y_hat):
  fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()  

print 'Writing finished!'



