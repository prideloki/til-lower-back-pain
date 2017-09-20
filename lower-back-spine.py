
# coding: utf-8

# In[1]:

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset
# 
# Prediction is done by using binary classification.
# 
# - Attribute1  = pelvic_incidence  (numeric)
# - Attribute2 = pelvic_tilt (numeric)
# - Attribute3 = lumbar_lordosis_angle (numeric)
# - Attribute4 = sacral_slope (numeric)
# - Attribute5 = pelvic_radius (numeric)
# - Attribute6 = degree_spondylolisthesis (numeric)
# - Attribute7 = pelvic_slope (numeric)
# - Attribute8 = Direct_tilt (numeric)
# - Attribute9 = thoracic_slope (numeric)
# - Attribute10 = cervical_tilt (numeric)
# - Attribute11 = sacrum_angle (numeric)
# - Attribute12 = scoliosis_slope (numeric)
# 
# Attribute class {Abnormal, Normal} 

# In[2]:

dataset = pd.read_csv('data/Dataset_spine.csv',usecols=[
    'Col1',
    'Col2',
    'Col3',
    'Col4',
    'Col5',
    'Col6',
    'Col7',
    'Col8',
    'Col9',
    'Col10',
    'Col11',
    'Col12',
    'Class_att'
])


# In[3]:

dataset.head()


# In[4]:

training_features = [
    'pelvic_incidence',
    'pelvic_tilt',
    'lumbar_lordosis_angle',
    'sacral_slope',
    'pelvic_radius',
    'degree_spondylolisthesis',
    'pelvic_slope',
    'direct_tilt',
    'thoracic_slope',
    'cervical_tilt',
    'sacrum_angle',
    'scoliosis_slope',
]
target = 'class_att'
dataset.columns = training_features + [target]


# In[5]:

dataset.info()


# In[6]:

dataset.head()


# ## EDA

# with regression line

# In[7]:

sns.set(style="ticks")
sns.pairplot(dataset, hue="class_att", kind="reg")
plt.show()


# In[8]:

sns.set(style="ticks")
sns.pairplot(dataset, hue="class_att")
plt.show()


# split training and test set

# In[9]:

train_x, test_x, train_y, test_y = train_test_split(dataset[training_features], dataset[target], test_size=0.2, random_state=42)


# In[10]:

print(train_x.shape)
print(train_y.shape)


# In[11]:

print(test_x.shape)
print(test_y.shape)


# Use logistic regression to predict the outcome

# In[25]:

model = LogisticRegression()

model.fit(train_x, train_y)


# In[27]:

y_pred = model.predict(test_x)


# In[28]:

y_pred


# calculating the accuracy

# In[29]:

model.score(test_x, test_y)


# In[ ]:



