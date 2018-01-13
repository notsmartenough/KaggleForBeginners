#import pandas for reading and manipulating datasets, numpy because we always need numpy
import pandas as pd
import numpy as np

#load the train and test sets to pandas dataframes, we have no header because of the output format of Kaggle
train = pd.read_csv('train.csv', header = None)
test = pd.read_csv('test.csv', header = None)

#Standard scaler allows us to scale (or normalize) the data - normalization makes our classification algo work correctly
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(train)
x_test = sc.transform(test)

#read the  predicted labels
y_train = pd.read_csv('trainLabels.csv', header = None)

#now for classification
import sklearn.ensemble as ske
randClf = ske.RandomForestClassifier(n_estimators = 100)
randClf.fit(x_train,y_train.values.ravel())

#predicted values
predictionSet = randClf.predict(x_test) 

#make the submission dataframe as required by Kaggle
submission = pd.DataFrame(data = {'Id':np.arange(1,9001), 'Solution':predictionSet})

#drop index because Kaggle output format doesn't need it
submission.index.drop

#save to a csv file for submission
submission.to_csv('submission1.csv', index = False)

