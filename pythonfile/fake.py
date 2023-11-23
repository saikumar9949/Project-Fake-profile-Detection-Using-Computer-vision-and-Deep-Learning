#importing essential librlibraries...
import sys
import csv
import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.show()
#Function for reading datasets from xls files 
def read_datasets():
  genuine_users = pd.read_csv("users.xls")
  fake_users = pd.read_csv("fusers.xls")
  x=pd.concat([genuine_users,fake_users ])
  y=len(fake_users)*[0]+len(genuine_users)*[1]
  return x,y
#Function for predicting sex using name of person
def predict_sex(name):
  name = str(name)
  sex_predictor = gender.Detector(case_sensitive=False)
  first_name = name.split(' ')[0]
  sex = sex_predictor.get_gender(first_name)
  sex_dict={'female': 2,'mostly_female': -1,'unknown': 0,'mostly_male': 1,'male': 2}
  sex_code = sex_dict[sex]
  return sex_code
#Function for feature engineering.. 
def extract_features(x):
  lang_list = list(enumerate(np.unique(x['lang'])))
  lang_dict = {name : i for i,name in lang_list}
  x.loc[:,'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
  x.loc[:,'sex_code']=predict_sex(x['name'])
  feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
  x = x.loc[:,feature_columns_to_use]
  return x
#Function for plotting learning curve...
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
  plt.figure()
  plt.title(title)
  if ylim is not None:
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores,axis=1)
  train_scores_std = np.std(train_scores,axis=1)
  train_scores_mean = np.mean(test_scores,axis=1)
  test_scores_std = np.std(test_scores,axis=1)
  plt.grid()
  plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
  plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
  plt.plot(train_sizes,train_scores_mean,'0-',color="r",label="Training score")
  plt.plot(train_sizes,test_score_mean,'0-',color="g",label="cross-validation score")
  plt.legend(loc="best")
  return plt 
#Function for plotting confusion matrix...  
def plot_confusion_matrix(cm,title='confusion matrix',cmap=plt.cm.Blues):
  target_names=['Fake','Genuine']
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arrange(len(target_names))
  plt.xticks(ticks_marks,target_names,rotation=45)
  plt.yticks(tick_marks,target_names)
  plt.tight_layout()
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
#Function for plotting ROC curve...
def plot_roc_curve(y_test,y_pred):
  false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_pred)
  print("False positive rate: ",false_positive_rate)
  print("True Positive rate: ",true_positive_rate)
  roc_auc = auc(false_positive_rate,true_positive_rate)
  plt.title('Receiver Operating Characteristic')
  plt.plot(false_positive_rate,true_positive_rate,b,label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.Xlim([-0.1,1.2])
  plt.ylim([-0.1,1.2])
  plt.ylabel('Trur Positive Rate')
  plt.Xlabel('False Positive Rate')
  plt.show()
#Function for training data using Support Vector Machine(SVM)..
def tarin(X_train,y_train,X_test):
  X_train=prepocessing.scale(X_train)
  X_test=preprocessing.scale(X_test)
  Cs = 10.0 ** np.arange(-2,3,.5)
  gammas = 10.0 ** np.arange(-2,3,.5)
  param = [{'gamma': gammas, 'C:' :Cs}]
  cvk = StratifiedKFold(y_train,n_folds=5)
  classifier = SVC()
  clf = GridSearchCV(classifier,param_grid=param,cv=cvk)
  clf .fit(X_train,y_train)
  print("The best classifier is: ",clf.best_estimator_)
  clf.best_estimator_.fit(X_train,y_train)
  scores = cross_validation.cross_val_score(clf.best_estimator_,X_train,y_train,cv=5)
  print(scores)
  print('Estimated score: %0.5f (+/- %0.5f' % (scores.mean(),scores.std()/2))
  title = 'Learning Curves (SVM,rbf kernel, $\gamma=%.of$)' %clf.best_estimator_.gamma
  plot_learning_curve(clf.best_estimator_,title,X_train,y_train,cv=5)
  plt.show()
  y_pred = clf.best_estimator_.predict(X_test)
  return y_test,y_pred
def train_randomforest(X_train,y_train,X_test):
  X_trsintrain = preprocessing.scale(X_train)
  X_test=preprocessing.scale(X_test)
  Cs = 10.0 ** np.arange(-2,3,.5)
  gammas = 10.0 ** np.arange(-2,3,.5)
  param = [{'gammas': gammas,'C': Cs}]
  cvk = StratifiedfeilfKFold(y_train,n_folds=5)
  classifier = SVC()
  clf = RandomForestClassfier()
  clf.flt(X_train,y_train)
  print("THe best classifier is : ",clf.best_estimator_)
  scores = cross_validation.cross_val_score(clf.best_estimator_,X_train,y_train,cv=5)
  print(scores)
  print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(),scores.std()/2))
  title = 'Learning curve (SVM,rbf kernel,$\gamma=%.61f$)' %clf.best_estimator_.gamma
  plot_learning_curve(clf.best_estimator_,title,X_train,y_train,cv=5)
  plt.show()
  y_pred = clf.best_estimator_.predict(X_test)
  return y_test,y_pred
print("Reading datasets......\n")
X,y= read_datasets()
print("extracting features....\n")
X = extract_features(X)
print(X.columns)
print(X.describe()) 
print("Spliting  datasets in train and test dataset....\n")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=44)
#RANDOM FOREST...
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_test,y_test)
result_rf=random_forest.score(X_test,y_test)

with open('rf.pkl','wb') as my_file_obj:
  pickle.dump(result_rf,my_file_obj)
print("File Stored Successfully")
cnf_matrix = confusion_matrix(y_test,y_pred)
print(cnf_matrix)
import seaborn as sns
labels = [0,1]
sns.heatmap(cnf_matrix, annot= True, cmap="jet",fmt = ".3f",xticklabels=labels,yticklabels=labels)
#Deep Neural Network.........
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential([
    Dense(units=16,input_dim=7,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
X_train = np.array(X_train)
y_train = np.array(y_train)
model.fit(X_train,y_train,batch_size=7,epochs=5)
X_test = np.array(X_test)
y_test = np.array(y_test)
score = model.evaluate(X_test,y_test)
print(score)
result_dnn=score[1]
print(result_dnn)
import pickle

with open('dnn.pkl','wb') as my_file_obj:
  pickle.dump(result_dnn,my_file_obj)
print("File Stored Successfully")
y_pred = model.predict(X_test)
Y_test = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_test,y_pred.round())
print(cnf_matrix)
import seaborn as sns
labels = [0,1]
sns.heatmap(cnf_matrix,annot=True,cmap="jet",fmt = ".3f",xticklabels=labels,yticklabels=labels)
