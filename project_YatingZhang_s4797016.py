#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter


# In[2]:


# concatenate train.csv and add_train.csv
df_t1 = pd.read_csv('train.csv')
df_t2 = pd.read_csv('add_train.csv')
df = pd.concat([df_t1, df_t2], ignore_index=True)
# print(df_t2.info())
# print(df_t1.info())
# print(df.info())


# In[3]:


# Step 1. split data train set/ test set


# In[4]:


y = df.iloc[:,-1].values  # label col
X_num = df.iloc[:,:100].values #numerical cols 
X_nom = df.iloc[:,100:-1].values #nominal cols 
X_num_train, X_num_test, X_nom_train, X_nom_test, y_train, y_test = train_test_split(X_num, X_nom, y, random_state = 0)

X_train = np.concatenate((X_num_train, X_nom_train), axis = 1) 
X_test = np.concatenate((X_num_test, X_nom_test), axis = 1)


# In[5]:


feature_column_names=list(df.columns[:128])
label_column_names=list(df.columns[-1:])

df_x_train=pd.DataFrame(X_train, columns=feature_column_names)
df_x_test=pd.DataFrame(X_test, columns=feature_column_names)

df_y_train=pd.DataFrame(y_train, columns=label_column_names)
df_y_test=pd.DataFrame(y_test, columns=label_column_names)

df_train=pd.concat([df_x_train, df_y_train], axis=1)
df_test=pd.concat([df_x_test, df_y_test], axis=1)

# print(df_train.describe().round(2)) # train set
# print(df_test.describe().round(2))  # test set
# print(df_test.info())   #25%
# print(df_train.info())  #75%


# In[6]:


# Step 2. imputation separately for train set/test set START


# In[7]:


# imputation for train set
#  train-numerical
df_train_impu = df_train.copy() 
df_train_impu.iloc[:,:100] = df_train_impu.iloc[:,:100].fillna(df_train_impu.iloc[:,:100].mean()) 

#  train-nominal
df_train_impu.iloc[:,-29:] = df_train_impu.iloc[:,-29:].fillna(df_train_impu.iloc[:,-29:].mode().iloc[0])
df_train=df_train_impu
y_train = df_train.iloc[:,-1].values  # label col
X_num_train = df_train.iloc[:,:100].values #numerical cols 
X_nom_train = df_train.iloc[:,100:-1].values #nominal cols 
# print(df_train.head(10))
print(df_train.info())
# print(X_nom_train)


# In[8]:


# imputation for test set
#  test-numerical
df_test_impu = df_test.copy() 
df_test_impu.iloc[:,:100] = df_test_impu.iloc[:,:100].fillna(df_test_impu.iloc[:,:100].mean()) 

#  test-nominal
df_test_impu.iloc[:,-29:] = df_test_impu.iloc[:,-29:].fillna(df_test_impu.iloc[:,-29:].mode().iloc[0])
df_test=df_test_impu
y_test = df_test.iloc[:,-1].values  # label col
X_num_test = df_test.iloc[:,:100].values #numerical cols 
X_nom_test = df_test.iloc[:,100:-1].values #nominal cols 
# print(df_test.head(10))
# print(X_nom_test)
print(df_test.info())


# In[9]:


# Step 3. normalisatin for numerical cols


# In[10]:


# standardization (or z-score normalization) numerical features
scaler = StandardScaler() 
scaler.fit(X_num_train) 
X_num_train = scaler.transform(X_num_train) 
X_num_test = scaler.transform(X_num_test)

X_train = np.concatenate((X_num_train, X_nom_train), axis = 1) 
X_test = np.concatenate((X_num_test, X_nom_test), axis = 1)


# In[11]:


# re-build data frame after normalisation
df_x_train=pd.DataFrame(X_train, columns=feature_column_names)
df_x_test=pd.DataFrame(X_test, columns=feature_column_names)

df_train=pd.concat([df_x_train, df_y_train], axis=1)
df_test=pd.concat([df_x_test, df_y_test], axis=1)

# print(df_train.describe().round(2)) # train set
# print(df_test.describe().round(2))  # test set
# print(df_test.info())


# In[12]:


# Step 4. One-hot encoding for nominal cols


# In[13]:


# for train set
nominal_columns = list(range(100, 128)) #nominal features from col_100 to col_127
# print(nominal_columns)

# for nominal features
ohe=OneHotEncoder()
encoded_features = pd.DataFrame() 
# feature_array = ohe.fit_transform(df_train.iloc[:,3].to_frame()).toarray()
for column_index in nominal_columns:
    # Encode the current (nominal) column
    feature_array = ohe.fit_transform(df_train.iloc[:, column_index].to_frame()).toarray()
    feature_labels = [f'col_{column_index}_{i}' for i in range(feature_array.shape[1])]
    encoded_column = pd.DataFrame(feature_array, columns=feature_labels)
    encoded_features = pd.concat([encoded_features, encoded_column], axis=1)

df_train_new=pd.concat([df_train.iloc[:,:100],encoded_features,df_train.iloc[:,[128]]],axis = 1)
df_train=df_train_new
# print(encoded_features.info()) 


# In[14]:


# for test set
# for nominal features
ohe=OneHotEncoder()
encoded_features = pd.DataFrame() 
feature_array = ohe.fit_transform(df_test.iloc[:,3].to_frame()).toarray()
for column_index in nominal_columns:
    # Encode the current (nominal) column
    feature_array = ohe.fit_transform(df_test.iloc[:, column_index].to_frame()).toarray()
    feature_labels = [f'col_{column_index}_{i}' for i in range(feature_array.shape[1])]
    encoded_column = pd.DataFrame(feature_array, columns=feature_labels)
    encoded_features = pd.concat([encoded_features, encoded_column], axis=1)

df_test_new=pd.concat([df_test.iloc[:,:100],encoded_features,df_test.iloc[:,[128]]],axis = 1)
df_test=df_test_new
# print(encoded_features.info()) 


# In[15]:


# print(df_train.head(10))
# print(df_train.info())

# print(df_test.head(10))
# print(df_test.info())


# In[16]:


# Pre-processing END
# Classifier START


# In[17]:


# Decision tree


# In[18]:


X_train = df_train.iloc[:,:186].values  #all the columns values except the label
y_train = df_train.iloc[:,-1:].values
y_train = y_train.ravel()   

X_test = df_test.iloc[:,:186].values  #all the columns values except the label
y_test = df_test.iloc[:,-1:].values


# In[19]:


dt = DecisionTreeClassifier(random_state = 0)
dt = dt.fit(X_train, y_train)


# In[20]:


y_pred_dt = dt.predict(X_test)  


# In[21]:


# evaluation - f1-score
f1_dt = metrics.f1_score(y_test, y_pred_dt, average='macro') 
print("The test macro f1-score of decision tree on the dataset is: ", f1_dt)


# In[22]:


# random forest


# In[23]:


rf = RandomForestClassifier(random_state=0) 
rf = rf.fit(X_train,y_train)


# In[24]:


y_pred_rf = rf.predict(X_test)
print(y_pred_rf)


# In[25]:


f1_rf = metrics.f1_score(y_test, y_pred_rf, average='macro') 
print("The test macro f1-score of random forest on the dataset is: ", f1_rf)


# In[26]:


# k-NN 5


# In[27]:


kNN_5 = KNeighborsClassifier(n_neighbors = 5)   # k defaule is 5


# In[28]:


kNN_5.fit(X_train, y_train)


# In[29]:


y_pred_kNN = kNN_5.predict(X_test)


# In[30]:


# when k=5，f1-score
f1_kNN = metrics.f1_score(y_test, y_pred_kNN, average='macro') 
print("The test macro f1-score of decision tree on the dataset is: ", f1_kNN)


# In[31]:


# find the best k value using CV
parameters = [{'n_neighbors': [int(x) for x in np.arange(1, 22, 2)]}]
kNN = KNeighborsClassifier()
clf_best_kNN = GridSearchCV(kNN, parameters, cv=5, scoring='f1_macro') 
clf_best_kNN.fit(X_train, y_train) 
print(clf_best_kNN.best_params_)


# In[32]:


# best k value is 17


# In[33]:


kNN_17 = KNeighborsClassifier(n_neighbors = 17) 
f1_kNN_17 = cross_val_score(kNN_17, X_train, y_train, cv=5, scoring=('f1_macro')) 
print("The cross-validation f1-score is:{:}".format( f1_kNN_17.mean()))


# In[34]:


kNN_17.fit(X_train, y_train)


# In[35]:


y_pred_kNN_17 = kNN_17.predict(X_test)


# In[36]:


f1_kNN_17 = metrics.f1_score(y_test, y_pred_kNN_17, average='macro') 
print("The test macro f1-score of decision tree on the dataset is: ", f1_kNN_17)


# In[37]:


# Naive Bayse 


# In[38]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[39]:


# 5-folder cross validation
acc_gnb = cross_val_score(gnb, X_train, y_train, cv=5, scoring=('accuracy')) 
f1_gnb = cross_val_score(gnb, X_train, y_train, cv=5, scoring=('f1_macro')) 
print("The cross-validation accuracy is: {:}\nThe cross-validation f1-score is:{:}".format(acc_gnb.mean(), f1_gnb.mean()))


# In[40]:


y_pred_gnb = gnb.predict(X_test)


# In[41]:


# f1-score
f1_gnb = metrics.f1_score(y_test, y_pred_gnb, average='macro') 
print("The test macro f1-score of decision tree on the dataset is: ", f1_gnb)


# In[42]:


y_pred_gnb = y_pred_gnb.astype(int)  # Convert the array to integers

# Create a DataFrame from the NumPy array
df_y_pred_gnb = pd.DataFrame({'Index': range(len(y_pred_gnb)), 'Predictions': y_pred_gnb})

# Save the DataFrame to a CSV file without the header
# df_y_pred_gnb.to_csv('NB_result_test.csv', header=False, index=False)


# In[43]:


# Calculate the CV accuracy and f1-score for the report

rf_accuracy = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
rf_f1 = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')

kNN_17_accuracy = cross_val_score(kNN_17, X_train, y_train, cv=5, scoring='accuracy')
kNN_17_f1 = cross_val_score(kNN_17, X_train, y_train, cv=5, scoring='f1_macro')

gnb_accuracy = cross_val_score(gnb, X_train, y_train, cv=5, scoring='accuracy')
gnb_f1 = cross_val_score(gnb, X_train, y_train, cv=5, scoring='f1_macro')

ensemble_accuracy = []
ensemble_f1 = []

for fold in range(5): # n folds=5
    y_pred_rf = cross_val_predict(rf, X_train, y_train, cv=5)
    y_pred_kNN_17 = cross_val_predict(kNN_17, X_train, y_train, cv=5)
    y_pred_gnb = cross_val_predict(gnb, X_train, y_train, cv=5)
    
    # Combine predictions (majority voting)
    ensemble_predictions = np.apply_along_axis(lambda x: int(np.median(x)), axis=0, arr=[y_pred_rf, y_pred_kNN_17, y_pred_gnb])

    # Calculate accuracy and F1-score for the ensemble within each fold
    ensemble_accuracy.append(accuracy_score(y_train, ensemble_predictions))
    ensemble_f1.append(f1_score(y_train, ensemble_predictions, average='macro'))

# Calculate the average accuracy and F1-score for the ensemble model, round to 3 decimals 
average_ensemble_accuracy = round(sum(ensemble_accuracy) / 5,3)
average_ensemble_f1 = round(sum(ensemble_f1) / 5,3)

print("Average Ensemble Accuracy:", average_ensemble_accuracy)
print("Average Ensemble F1-Score:", average_ensemble_f1)


# In[44]:


# Training END


# In[45]:


# Prediction START 


# In[46]:


df_ftest= pd.read_csv('test.csv')
# print(df_ftest.info())
# print(df_ftest.describe())
# y_ftest = df_ftest.iloc[:,-1].values  # label col
X_num_ftest = df_ftest.iloc[:,:100].values #numerical cols 
X_nom_ftest = df_ftest.iloc[:,100:].values #nominal cols 


# In[47]:


# pre-processing method same with training: normalisatin for numerical cols; one-hot encoding for nominal cols.

# 1. standardization numerical features
# scaler = StandardScaler() 
# scaler.fit(X_num_train) 
X_num_ftest = scaler.transform(X_num_ftest)
# train set and test set's standardisation are all baseed on train set.
X_ftest = np.concatenate((X_num_ftest, X_nom_ftest), axis = 1)
print(len(X_ftest))
print(X_ftest.shape[1])

df_x_ftest=pd.DataFrame(X_ftest, columns=feature_column_names)
# df_ftest=pd.concat([df_x_ftest, df_y_test], axis=1)


# In[48]:


# 2. one-hot encoding for nominal cols.
# for nominal features
nominal_columns_f = list(range(100, 128)) #nominal features from col_100 to col_127

ohe_f=OneHotEncoder()
encoded_features_f = pd.DataFrame() 
for column_index in nominal_columns_f:
    # Encode the current (nominal) column
    feature_array_f = ohe_f.fit_transform(df_x_ftest.iloc[:, column_index].to_frame()).toarray()
    feature_labels_f = [f'col_{column_index}_{i}' for i in range(feature_array_f.shape[1])]
    encoded_column_f = pd.DataFrame(feature_array_f, columns=feature_labels_f)
    encoded_features_f = pd.concat([encoded_features_f, encoded_column_f], axis=1)

df_x_ftest=pd.concat([df_x_ftest.iloc[:,:100],encoded_features_f],axis = 1)
# print(encoded_features_f.info()) 


# In[49]:


# print(df_x_ftest.head(10))
# print(df_x_ftest.info())


# In[50]:


x_ftest = df_x_ftest.iloc[:,:186].values  #all the columns values except the label
# print(x_ftest)


# In[51]:


#decision tree
dt_result = dt.predict(x_ftest)
# print(dt_result)


# In[52]:


# random forest 
rf_result = rf.predict(x_ftest)
y_pred_rf = rf.predict(X_test)
# print(rf_result)
# print(y_pred_rf)


# In[53]:


# k17-nn
knn17_result = kNN_17.predict(x_ftest)
# print(knn17_result)


# In[54]:


# naive bayse
nb_result = gnb.predict(x_ftest)
# print(nb_result)


# In[55]:


# based on the f1-score, choose random forest, k17-nn, and naive bayse for assembling 


# In[56]:


# stack the three results (using numpy.vstack)
stacked_results = np.vstack((rf_result, knn17_result, nb_result))

# Find the mode (most frequent element)
most_common = [Counter(column).most_common(1)[0][0] for column in stacked_results.T]
# print(most_common)


# In[57]:


# # export random forest results
# df_rf_result=pd.DataFrame(rf_result)
# df_rf_result.to_csv('df_rf_result.csv', header=False, index=True, )
# # export knn-17 results
# df_knn17_result=pd.DataFrame(knn17_result)
# df_knn17_result.to_csv('df_knn17_result.csv', header=False, index=True, )
# # export naive bayse results
# df_nb_result=pd.DataFrame(nb_result)
# df_nb_result.to_csv('df_nb_result.csv', header=False, index=True, )
# # stacked results
# df_stacked_result=pd.DataFrame(stacked_results)
# df_stacked_result.to_csv('df_stacked_result.csv', header=False, index=True, )

# export the final result
most_common = [int(item) for item in most_common]  # Convert the most_common (from array) to integer 
df_final_result=pd.DataFrame(most_common)
# df_final_result.reset_index(level=0, inplace=True)
new_row = pd.DataFrame({'index': ["0.972"], 0: ["0.979"]})
df_final_result = pd.concat([df_final_result, new_row], ignore_index=True,axis=0)
df_final_result.to_csv('s4797016.csv', header=False, index=False, )


# In[58]:


# print(df_final_result.info)


# In[59]:


# reference
# INFS7202 teaching team, the INFS7203 course codeing guide.

