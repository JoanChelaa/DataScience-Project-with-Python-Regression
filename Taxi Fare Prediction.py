#!/usr/bin/env python
# coding: utf-8

# # Setting up the environment

# In[1]:


#Pandas will help read the csv files 
#Python OS module provides the facility to establish the interaction between the user and the operating system
#numpy helps as work with arrays.

import numpy as np
import os 
import pandas as pd
from pyarrow import feather
from tqdm import tqdm


#print a list containing the names of all the entries in the directory given path
# Open a file
path= '../data science project'
dirs= os.listdir( path )
# This would print all the files and directories
for file in dirs:
   print(file)


# # Setting up the training environment

# In[2]:


training_path = '../data science project/train.csv'
 
    
#We want to establish the exact number of rows in the training data

with open(training_path) as file:
    n_rows = len(file.readlines())
print (f'Exact number of rows: {n_rows}')


# In[7]:


#Have a look at the training file header

df_tmp = pd.read_csv(training_path, nrows=5)
df_tmp.head()


# In[8]:


df_tmp.info()


# In[11]:


chunksize = 10_000_000 #10 million rows at a go

#To be able to hold the batch dataframe
df_list = []

for df_chunk in tqdm(pd.read_csv(training_path, usecols=cols, dtype=traintypes, chunksize=chunksize)):
    df_list.append(df_chunk)


# In[12]:


# We need to optimize memory usage by setting up the columns to the most suitable type
traintypes = {'fare_amount': 'float32',
            'pickup_datetime': 'string',
            'pickup_longitude': 'float32',
            'pickup_latitude': 'float32',
             'dropoff_longitude':'float32',
             'dropoff_latitude':'float32',
             'passenger_count': 'uint8'}
cols = list(traintypes.keys())


# In[13]:


# Merging all dataframes into one
train_df = pd.concat(df_list)

#To release the memory we delete the dataframe
del df_list

#To be able to see what has been loaded
train_df.info()


# In[14]:


display(train_df.head())
display(train_df.tail())


# ### We can save the file into a feather format which will allow us to read the same dataframe next time directly without reading the csv file again

# In[15]:


#The feather format is allowed by importing pyarrow and using the feather module
train_df.to_feather('nyc_taxi_data_raw.feather')


# In[16]:


train_df_new = pd.read_feather('nyc_taxi_data_raw.feather')


# In[17]:


# To establish that we have loaded the 55 million rows we print the dataframe

train_df_new.info()


# In[18]:


train_df_new.info


# In[19]:


# Add features to the dataframe
#The travel vector represents the pickup location to the dropff location
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    
add_travel_vector_features(train_df_new)


# # Data cleaning- Pruning Outliers

# In[20]:


print(train_df_new.isnull().sum())


# In[21]:


#Removing observations with missing values
print('Old size: %d' %len(train_df_new))
train_df_new = train_df_new.dropna(how ='any', axis='rows')
print('New size: %d' %len(train_df_new))

# Removing observations with erroneous values
mask = train_df_new['pickup_longitude'].between(-75, -73)
mask &= train_df_new['dropoff_longitude'].between(-75, -73)
mask &= train_df_new['pickup_latitude'].between(40, 42)
mask &= train_df_new['dropoff_latitude'].between(40, 42)
mask &= train_df_new['passenger_count'].between(0, 8)
mask &= train_df_new['fare_amount'].between(0, 250)

train_df_new = train_df_new[mask]
print('Error free: %d' %len(train_df_new))


# In[22]:


#Distribution of our subset travel vector
plot = train_df_new.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude') 


# In[23]:


print('Old size: %d' % len(train_df_new))
train_df_new = train_df_new[(train_df_new.abs_diff_longitude < 5.0) & (train_df_new.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df_new))


# # Training the model

# In[24]:


#Establish an input matrix for our linear model.
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))
train_X = get_input_matrix(train_df_new)
train_Y = np.array(train_df_new['fare_amount'])

print(train_X.shape)
print(train_Y.shape)


# In[25]:


# We seek to find the optimal weight column using the Ordinary Least Squares -numpy used.
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_Y)
print(w_OLS)


# # Evaluating the model using the test set
# 
# 

# In[26]:


testing_df = pd.read_csv('../data science project/test.csv')
testing_df.dtypes


# In[27]:


# We need to generate the input matrix by adding our features to the above helper functions.
add_travel_vector_features(testing_df)
X_test = get_input_matrix(testing_df)


#Proceed to make predictions of the fare_amount on the test set based on our model (weight column) developed using the training set
Y_test_predictions = np.matmul(X_test,w_OLS).round(decimals = 2)

#We can load our predictions as csv files
load=pd.DataFrame({'key': testing_df.index, 'fare_amount': Y_test_predictions},
    columns = ['key', 'fare_amount'])
load.to_csv('load.csv', index = False)

print(os.listdir('.'))


# In[ ]:





# In[ ]:





# In[ ]:




