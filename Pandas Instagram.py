#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# In[2]:


Insta_db = pd.read_csv(r"F:\New folder\Instagram.csv")


# In[3]:


Insta_db


# In[4]:


Insta_db.head()


# In[5]:


Insta_db.info() #check the information about data


# In[6]:


df = pd.DataFrame(Insta_db )
print(df)


# In[7]:


df.describe() #describe the data set shows the max min mean


# In[8]:


df.isna() #show the null value in boolen


# In[9]:


df.isna().sum() #how many null value present in there data set


# In[10]:


df.shape


# In[28]:


df.columns


# In[29]:


len(df.columns)


# In[11]:


df.isna() #here true means null values in the data


# In[12]:


df.fillna(0,inplace= True)


# In[13]:


df


# In[14]:


df.nunique() #identifying unique values from the dataframe


# In[15]:


#	Drop unnecessary fields
df.drop('Caption', axis=1, inplace=True) 


# In[16]:


#rename columns
df.rename(columns ={'From Home': 'From_Home'} , inplace = True)
df.rename(columns ={'From Hashtags': 'From_Hashtags'} , inplace = True)
df.rename(columns ={'From Explore': 'From_Explore'} , inplace = True)
df.rename(columns ={'From Other': 'From_Other'} , inplace = True)
df.rename(columns ={'Profile Visits': 'Profile_Visits'} , inplace = True)


# In[17]:


df.info() #checking for the drop column and Rename


# In[18]:


#checking outlier in From Hashtags column
y = list(df.From_Hashtags) 
sns.boxplot(y)


# In[19]:


#checking outlier in Likes column
a = list(df.Likes) 
sns.boxplot(a)


# In[20]:


sns.set(style = 'whitegrid')
#plt.figure(figsize=(25,10))
sns.violinplot(x ="Likes",y ="Impressions",data = df)


# In[21]:


plt.figure(figsize=(15, 8))
plt.title("Distribution of Impressions from Home")
sns.barplot(x='From_Home',y='Impressions',data=df)
plt.show()


# In[22]:


home = df["From_Home"].sum()
hashtags = df["From_Hashtags"].sum()
explore = df["From_Explore"].sum()
other = df["From_Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(df, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


# In[23]:


figure = px.scatter(data_frame = df, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# Analyzing Conversion Rate
# In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. The formula that you can use to calculate conversion rate is (Follows/Profile Visits) * 100. Now let’s have a look at the conversation rate of my Instagram account:

# In[24]:


conversion_rate = (df["Follows"].sum() / df["Profile_Visits"].sum()) * 100
print(conversion_rate)


# # Instagram Reach Prediction Model

# In[25]:


x = np.array(df[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile_Visits', 'Follows']])
y = np.array(df["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=42)


# In[26]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[27]:


#features = [['Likes','Saves', 'Comments', 'Shares', 'Profile_Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)

