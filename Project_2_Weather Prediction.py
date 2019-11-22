#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectFdr, f_classif, mutual_info_classif, chi2, RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer, LabelBinarizer, OneHotEncoder,OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, RidgeClassifier, LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgbm
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes = True, style = 'ticks', palette = 'Set2')
pd.options.display.max_columns = 1000


# In[2]:


os.listdir()


# In[3]:


weather = pd.read_csv('weatherAUS.csv')


# In[5]:


weather.shape


# In[6]:


weather.isna().sum()


# In[7]:


weather['RainTomorrow'] = LabelEncoder().fit_transform(weather['RainTomorrow'])


# In[8]:


sns.heatmap(data= weather.isna(), cmap = 'viridis')


# In[9]:


#drop nan value in RainToday
weather = weather[~weather.RainToday.isna()]
display(weather.head())
weather.shape


# In[11]:


# weather = create_RainYesturady(weather)
RainYesturady = np.array(weather['RainToday'])
RainYesturady = np.insert(RainYesturady,0,3)
RainYesturady = np.delete(RainYesturady,-1)


# In[12]:


weather.insert((len(weather.columns)-1),'RainYesturady', RainYesturady)


# In[13]:


weather.drop(index=0, inplace = True)


# In[14]:


RainYesturady.shape


# In[15]:


display(weather.head(10))
weather.shape


# # Feature Selection

# In[25]:


weather_for_selection = weather.drop(['RISK_MM'], axis = 1)


# In[26]:


weather_for_selection.shape


# In[27]:


#change date to month
weather_for_selection['Date'] = weather_for_selection['Date'].astype('datetime64')
weather_for_selection['Date'] = weather_for_selection['Date'].dt.month
weather_for_selection.rename(columns = {'Date':'Month'}, inplace = True)


# In[28]:


remove_all_nan = weather_for_selection[~weather_for_selection.isna()]


# In[29]:


#split features into two lists based on their datatype
def split_col(dataframe):
    categorical_features = []
    numerical_features = []
    for i in dataframe.columns.values:
        if dataframe[i].dtypes != 'object':
            numerical_features.append(i)
        else:
            categorical_features.append(i)
    return categorical_features, numerical_features


# In[30]:


display(weather_for_selection.head())
category_f, numerical_f = split_col(remove_all_nan.iloc[:,:-1])


# ### Look at the replationship between each feature and the target variable

# In[31]:


def selection_catergory(category_f):
    result = []
    for i in np.arange(len(category_f)):
        x = remove_all_nan[~remove_all_nan[category_f[i]].isna()]
        feature = LabelEncoder().fit_transform(x[category_f[i]])
        label = x['RainTomorrow']
        fstat, pval = chi2(feature.reshape(-1,1), label)
        mi = mutual_info_classif(feature.reshape(-1,1), label)
        result.append([category_f[i], round(fstat[0],5), round(pval[0],5), round(mi[0],5)])
    return pd.DataFrame(result, columns =['Category_f', 'Chi2', 'Pval', 'MI'])

def selection_number(numerical_f):
    result = []
    for i in np.arange(len(numerical_f)):
        x = remove_all_nan[~remove_all_nan[numerical_f[i]].isna()]
        feature = StandardScaler().fit_transform(x[[numerical_f[i]]])
        label = x['RainTomorrow']
        fstat, pval = f_classif(feature.reshape(-1,1), label)
        mi = mutual_info_classif(feature.reshape(-1,1), label)
        result.append([numerical_f[i], round(fstat[0],5), round(pval[0],5), round(mi[0],5)])
    return pd.DataFrame(result, columns =['Number_f', 'Fstat', 'Pval', 'MI'])


# In[32]:


result1 = selection_catergory(category_f)
result2 = selection_number(numerical_f)
display(result1)
display(result2)


# In[133]:


fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
ax.plot(result1['Category_f'], result1['Pval'], label = 'Category_Pval', linewidth = 3)
ax.plot(result1['Category_f'], [0.05]*6, linestyle = '--', label = 'Pval_Benchmark', linewidth = 3)
plt.tick_params(axis = 'x',rotation = 30)
plt.xlabel('Categorical_features')
plt.ylabel('P-value')
plt.legend()

fig2 = plt.figure(figsize = (20,10))
ax2 = fig2.add_subplot(111)
ax2.plot(result2['Number_f'], result2['Pval'],label = 'Number_Pval', linewidth = 3)
ax2.plot(result2['Number_f'], [0.05]*len(result2['Number_f']), linestyle = '--', label = 'Pval_Benchmark', linewidth = 3)
plt.tick_params(axis = 'x',rotation = 30)
plt.xlabel('Numerical_features')
plt.ylabel('P-value')
plt.legend()


# #### Conclusion: Drop 'Month' which has relatively higher value, although its p-value is less than 0.05

# In[34]:


import sklearn
sklearn.metrics.SCORERS.keys()


# # First Trial: Remove all rows with nan values

# In[609]:


display(weather_for_selection.head())
weather_first_trial = weather_for_selection.dropna()
weather_first_trial.drop(['Month'], axis = 1, inplace = True)
display(weather_first_trial.head())
weather_first_trial.shape


# In[610]:


result2['Number_f'].values[1:]


# In[611]:


x_train, x_test, y_train, y_test = train_test_split(weather_first_trial.iloc[:,:-1], weather_first_trial.iloc[:,-1]
                                                    , test_size = 0.2, random_state = 300)

col = ColumnTransformer(transformers=[('standardized', StandardScaler(), result2['Number_f'].values[1:])],remainder = 'passthrough', 
                       sparse_threshold = 0)

x_test1 = x_test.copy()
y_test1 = y_test.copy()
#Resampling
x_train = pd.get_dummies(x_train, drop_first = True)
x_test = pd.get_dummies(x_test, drop_first = True)
x_train_columns = x_train.columns.values
x_train = col.fit_transform(x_train)
x_test = col.transform(x_test)
x_train, y_train = SMOTE(sampling_strategy='minority').fit_resample(x_train, y_train)
# neg_pos = 35220.0/9916.0

#models
logisticR = LogisticRegression(penalty = 'l2', solver = 'newton-cg', multi_class = 'multinomial',random_state = 300)
randomF = RandomForestClassifier(random_state=300)
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = 1),n_estimators=50, random_state=300)
xgboost = XGBClassifier(random_state=300)


# In[612]:


x_train_columns


# In[613]:


get_ipython().run_cell_magic('time', '', "\ndef check_model(models, models_name):\n    result_positive = []\n    for model in range(len(models)):\n        print(f'Processing: {models_names[model]}')\n        pipeline1 = Pipeline(steps=[('model', models[model])])\n        pipeline1.fit(x_train, y_train)\n        y_predicted = pipeline1.predict(x_test)\n        f1score = f1_score(y_test, y_predicted,average = None)[1]\n        result_positive.append([models_names[model], round(f1score,4)])\n    return pd.DataFrame(result_positive, columns = ['Model_Name', 'f1_Score_Positive'])")


# ### Compare 4 models' performance

# In[614]:


get_ipython().run_cell_magic('time', '', "models = [logisticR, randomF, adaboost, xgboost]\nmodels_names = ['logisticR', 'randomF', 'adaboost', 'xgboost']\nresult_positive = check_model(models, models_names)\nresult_positive")


# In[615]:


plt.figure(figsize = (20,10))
plt.plot(result_positive['Model_Name'], result_positive['f1_Score_Positive'], label = 'Model_Result', linewidth = 3)
plt.legend(loc =5)
plt.xlabel('Model_Name')
plt.ylabel('f1_Score_Positive')
plt.title('Model Selection')


# ## Final Model

# In[616]:


weather_first_trial['RainTomorrow'].value_counts()


# In[617]:


# neg_pos = 35220.0/9916.0
xgboost1 = XGBClassifier(n_estimators=300,random_state=300, max_depth = 5, cv = 3)
xgboost1.fit(x_train, y_train)
y_predicted = xgboost1.predict(x_test)

print(classification_report(y_test, y_predicted, target_names=['No', 'Yes']))

print(f1_score(y_test, y_predicted,average = None)[1])
cvs = cross_val_score(estimator=xgboost1, X=x_train, y = y_train,cv = 10, verbose=2,scoring='accuracy')
print(f'10-fold Cross Validation: {cvs.mean()}')


# In[462]:


fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111)
plot_confusion =confusion_matrix(y_test, y_predicted)
sns.heatmap(plot_confusion, annot = True, xticklabels = ['No', 'Yes'], yticklabels = ['No', 'Yes'])
plt.xlabel('Prediction')
plt.ylabel('Actual')


# In[463]:


fig2 = plt.figure(figsize = (10,15))
ax = fig2.add_subplot(111)
plot_importance(xgboost1, max_num_features = 10, ax=ax)


# In[495]:


features = weather_first_trial.columns[:-1]
print(list(features))
def prediction (data):
    data = pd.get_dummies(data, drop_first = True)
    data = col.fit_transform(data)
    print(xgboost1.predict(data))
    
Location = list(weather_first_trial['Location'].unique())
WindGustDir = list(weather_first_trial['WindGustDir'].unique())
WindDir9am = list(weather_first_trial['WindDir9am'].unique())
WindDir3pm = list(weather_first_trial['WindDir3pm'].unique())
print('------------------------')
print(f'Location:\n{Location}')
print('------------------------')
print(f'WindGustDir:\n{WindGustDir}')
print('------------------------')
print(f'WindDir9am:\n{WindDir9am}')
print('------------------------')
print(f'WindDir3pm:\n{WindDir3pm}')

# ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
# 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
# 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
# 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainYesturady']
#input a list of all feature above to below function
prediction()


# # Select the top 6 features (based on the feature importances above)

# In[344]:


selection = x_train_columns[[9,5,11,13,4,2]]
# [9,5,11,13,4,2]
selection


# In[345]:


x_weather_selected = weather_first_trial[selection]
y_weather_selected = weather_first_trial.iloc[:,-1]
display(x_weather_selected.head())
display(y_weather_selected.head())


# In[437]:


x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_weather_selected, y_weather_selected, 
                                                           test_size = 0.2, random_state = 200)
coltrans = ColumnTransformer(transformers = [('standard', StandardScaler(), selection)])
NegoverPos = y_train_s.value_counts()[0] / y_train_s.value_counts()[1]
x_train_s = coltrans.fit_transform(x_train_s)
x_test_s = coltrans.transform(x_test_s)

def determine_n_estimators(start = 100, end = 110, step = 1, **kwarg):
    result = []
    for i in range(start,end+1,step):
        xgboost2 = XGBClassifier(random_state=300,
                                 n_estimators=i,
                                 max_depth = 3,
                                 learning_rate=0.1, 
                                 min_child_weight = 1, 
                                 gamma = 0.2,
                                 subsample = 0.8,
                                 colsample_bytree= 0.8,
                                 objective= 'binary:logistic',
                                 scale_pos_weight = NegoverPos,
                                 n_jobs=-1)

        xgboost2.fit(x_train_s, y_train_s)
        y_predicted1 = xgboost2.predict(x_test_s)
        result.append([i,f1_score(y_test_s, y_predicted1,average = None)[1]])
    print(xgboost2.get_params)
    plotgraph = pd.DataFrame(result, columns = ['n_estimators', 'f1_score for +ve result'])
    sns.lineplot(data = plotgraph, x = 'n_estimators', y = 'f1_score for +ve result')
    plt.title(f'n_estimators from {start} to {end}, step = {step}')
    print(plotgraph.sort_values('f1_score for +ve result', ascending = False).iloc[0])
    return result


# In[438]:


determine_n_estimators(start = 100, end = 1000, step = 100)


# # Tuning Hyperparameters (I only kept the last turning and update the hyperparameter to the function above)

# In[440]:


xgboost2 = XGBClassifier(random_state=300,
                         n_estimators=900,
                         max_depth = 3,
                         learning_rate=0.1, 
                         min_child_weight = 1, 
                         gamma = 0.2,
                         subsample = 0.8,
                         colsample_bytree= 0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight = NegoverPos,
                         n_jobs=-1)

param2 = {'subsample':(0.6,0.7,0.8),
         'colsample_bytree':(0.6,0.7,0.8)}
gridsearch2 = GridSearchCV(estimator = xgboost2, param_grid=param2, cv = 3, scoring='f1')
gridsearch2.fit(x_train_s, y_train_s)
y_predicted1 = gridsearch2.predict(x_test_s)
print(classification_report(y_test_s, y_predicted1, target_names=['No', 'Yes']))


# In[441]:


gridsearch2.best_params_


# In[442]:


gridsearch2.best_score_


# In[376]:


plot_importance(xgboost2)


# # Second Trial

# In[2]:


weather2 = pd.read_csv('weatherAUS.csv')


# In[3]:


weather2.shape


# In[4]:


weather2.isna().sum()


# In[5]:


weather2.drop(['RISK_MM', 'Date'],axis =1, inplace = True)


# ### Find Features with higher correlations

# In[6]:


weather2['RainTomorrow'] = LabelBinarizer().fit_transform(weather2.RainTomorrow)
corr = weather2.corr(method = 'pearson')
corr


# In[7]:


new_weather = weather2.drop(['MinTemp', 'MaxTemp','Evaporation', 'WindSpeed9am', 'WindSpeed3pm', 'Temp9am', 'Temp3pm', 'Pressure3pm', 'WindGustSpeed'], axis =1)
# display(new_weather[(abs(corr['Sunshine']) >= 0.5) & (abs(corr['Sunshine']) < 1)])
display(new_weather.head())
corr
#combine sunshine + humidity9pm + humidity3pm + cloud9am + cloud3pm
#pressure9am + pressure 3pm
ls = corr.drop(['MinTemp', 'MaxTemp','Evaporation', 'WindSpeed9am', 'WindSpeed3pm', 'Temp9am', 'Temp3pm'], axis =1)


# In[8]:


weight = corr.loc['RainTomorrow',['Sunshine','Humidity9am','Humidity3pm','Cloud9am','Cloud3pm']].reset_index()
weight


# In[9]:


new_weather.head()


# In[10]:


minmax = MinMaxScaler(feature_range = (0,1))
columntransformer = ColumnTransformer(transformers = [('ordinal', OrdinalEncoder(),['Location', 'WindGustDir','WindDir9am',
                                                                                   'WindDir3pm', 'RainToday'])],remainder = 'passthrough', sparse_threshold = 0)
new_weather.dropna(inplace = True)


# In[11]:


x_train_2nd, x_test_2nd, y_train_2nd, y_test_2nd = train_test_split(new_weather.iloc[:,:-1], new_weather.iloc[:,-1],
                                                                   test_size = 0.2, random_state = 222)


# In[12]:


x_train_2nd.head()


# ### Combine feartures with higher correlation

# In[13]:


numerical_features = ['Rainfall', 'Sunshine','Humidity9am','Humidity3pm','Pressure9am', 'Cloud9am','Cloud3pm']
#minmax the numerical_features
minmax =  MinMaxScaler(feature_range=(0, 1))
x_train_2nd[numerical_features] = minmax.fit_transform(x_train_2nd[numerical_features])
x_test_2nd[numerical_features] = minmax.transform(x_test_2nd[numerical_features])
display(x_train_2nd.head())
x_train_2nd['Sunshine_Humidity_Cloudy'] = x_train_2nd['Sunshine'] * weight.iloc[0,1] + x_train_2nd['Humidity9am'] * weight.iloc[1,1] + x_train_2nd['Humidity3pm'] * weight.iloc[2,1] + x_train_2nd['Cloud9am'] * weight.iloc[3,1] + x_train_2nd['Cloud3pm'] * weight.iloc[4,1]
# x_train_2nd['Pressure9am3pm'] = x_train_2nd['Pressure9am'] * weight.iloc[5,1] + x_train_2nd['Pressure3pm'] * weight.iloc[6,1] 

x_test_2nd['Sunshine_Humidity_Cloudy'] = x_test_2nd['Sunshine'] * weight.iloc[0,1] + x_test_2nd['Humidity9am'] * weight.iloc[1,1] + x_test_2nd['Humidity3pm'] * weight.iloc[2,1] + x_test_2nd['Cloud9am'] * weight.iloc[3,1] + x_test_2nd['Cloud3pm'] * weight.iloc[4,1]
# x_test_2nd['Pressure9am3pm'] = x_test_2nd['Pressure9am'] * weight.iloc[5,1] + x_test_2nd['Pressure3pm'] * weight.iloc[6,1] 

x_train_2nd.drop(weight['index'].values, axis = 1, inplace = True)
x_test_2nd.drop(weight['index'].values, axis = 1, inplace = True)
display(x_train_2nd.head())


# In[14]:


test = x_train_2nd.copy()
test['RainT'] = y_train_2nd
test.corr()


# In[15]:


# x_train_2nd['Sunshine_Humidity_Cloudy_WindGustSpeed'] = x_train_2nd['Sunshine_Humidity_Cloudy'] * 0.481937 + x_train_2nd['WindGustSpeed'] * 0.237568

# x_test_2nd['Sunshine_Humidity_Cloudy_WindGustSpeed'] =  x_test_2nd['Sunshine_Humidity_Cloudy'] * 0.481937 + x_test_2nd['WindGustSpeed'] * 0.237568

# x_train_2nd.drop(['WindGustSpeed','Sunshine_Humidity_Cloudy'], axis = 1, inplace = True)
# x_test_2nd.drop(['WindGustSpeed','Sunshine_Humidity_Cloudy'], axis = 1, inplace = True)
# x_train_2nd.head()


# In[16]:


x_train_2nd = columntransformer.fit_transform(x_train_2nd)
x_test_2nd = columntransformer.fit_transform(x_test_2nd)
x_train_2nd, y_train_2nd = SMOTE(sampling_strategy='minority').fit_resample(x_train_2nd, y_train_2nd)


# In[17]:


y_train_2nd.tolist().count(1)


# In[18]:


x_train_2nd.shape


# In[19]:


xgboost_2nd = XGBClassifier(n_estimators=350, learning_rate=0.1, max_depth=5, random_state=222 )
xgboost_2nd.fit(x_train_2nd, y_train_2nd)


# In[20]:


predicted_y_2nd = xgboost_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd,predicted_y_2nd))


# In[21]:


plot_importance(xgboost_2nd)


# ## Remove the least important feature

# In[22]:


x_train_2nd = np.delete(x_train_2nd, 4, axis = 1)
x_test_2nd = np.delete(x_test_2nd, 4, axis = 1)


# In[23]:


x_train_2nd.shape


# In[68]:


xgboost_2nd = XGBClassifier(random_state=300,
                         n_estimators=100,
                         max_depth = 9,
                         learning_rate=0.1, 
                         min_child_weight = 2, 
                         gamma = 0.1,
                         reg_alpha= 0.01,
                         subsample = 0.9,
                         colsample_bytree= 0.85,
                         objective= 'binary:logistic',
                         n_jobs=-1)


# In[70]:


param_2nd = {'n_estimators':np.arange(100,1000,100)}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1')
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))
# plot_importance(xgboost_2nd)



# In[71]:


gridsearch_2nd.best_params_


# In[72]:


gridsearch_2nd.best_estimator_


# ### 1st tuning

# In[36]:


param_2nd = { 'max_depth':(8,9,10), 'min_child_weight':(2,3,4)}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1', verbose=2)
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[37]:


gridsearch_2nd.best_params_


# In[39]:


gridsearch_2nd.best_estimator_


# ### 2rd Tuning

# In[46]:


param_2nd = {'gamma':[0.05, 0.1, 0.15]}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1', verbose=2)
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[47]:


gridsearch_2nd.best_params_


# In[48]:


gridsearch_2nd.best_estimator_


# ### 3nd Tuning

# In[57]:


param_2nd = {  'subsample':[0.80,0.85, 0.9, 0.95],
 'colsample_bytree':[0.70, 0.75, 0.80, 0.85, 0.90]}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1', verbose=2)
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[58]:


gridsearch_2nd.best_params_


# In[59]:


gridsearch_2nd.best_estimator_


# ### 4th Tuning

# In[61]:


param_2nd = { 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1', verbose=2)
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[63]:


print(gridsearch_2nd.best_params_)
gridsearch_2nd.best_score_


# In[67]:


param_2nd = {'n_estimators':np.arange(100,1000,100)}
gridsearch_2nd = GridSearchCV(estimator = xgboost_2nd, param_grid=param_2nd, cv = 3, scoring='f1', verbose=2,n_jobs=-1)
gridsearch_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = gridsearch_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[66]:


print(gridsearch_2nd.best_params_)
gridsearch_2nd.best_estimator_


# In[73]:


xgboost_2nd = XGBClassifier(random_state=300,
                         n_estimators=100,
                         max_depth = 9,
                         learning_rate=0.1, 
                         min_child_weight = 2, 
                         gamma = 0.1,
                         reg_alpha= 0.01,
                         subsample = 0.9,
                         colsample_bytree= 0.85,
                         objective= 'binary:logistic',
                         n_jobs=-1)
xgboost_2nd.fit(x_train_2nd, y_train_2nd)

predicted_y_2nd = xgboost_2nd.predict(x_test_2nd)
print(classification_report(y_test_2nd, predicted_y_2nd))


# In[74]:


plot_importance(xgboost_2nd)


# Further Study
# - can investigate more about features which are highly correlated
# - can try to use prediction models to predict the missing values
# - can create new features based on domain knowledge

<<<<<<< HEAD
# In[78]:


get_ipython().system('jupyter nbconvert --to script Project_2_basic_ver.ipynb')


# In[ ]:
=======
>>>>>>> d9118a5dbcf2c2483929bb9b1cd3cd620097ed8f




