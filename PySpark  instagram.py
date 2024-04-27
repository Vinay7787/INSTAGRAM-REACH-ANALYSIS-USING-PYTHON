#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyspark 


# ###### Step 1:Ceration of Spark Session and its its provide spark's functionality with a lesser number of constructs

# In[2]:


from pyspark.sql import SparkSession


# ##### Step 2:Creation of DataFrameApp using Spark Session

# In[3]:


ss=SparkSession.builder.appName("DataFrameApp").getOrCreate()


# ##### Step 3:Load dataset

# In[4]:


Insta= ss.read.option('header','True').csv(r"F:\New folder\Instagram.csv")


# In[5]:


Insta.show(5)


# In[6]:


Insta.printSchema() #info of Schema


# In[7]:


Insta.tail(2)# its shows last 2 entry


# In[8]:


Insta.columns #its shows the names of columns


# In[9]:


insta=Insta.withColumnRenamed("Impressions","impressions").withColumnRenamed("From Home","fhome").withColumnRenamed("From Hashtags","fhashtags").withColumnRenamed("From Explore","fe").withColumnRenamed("From Other","fo").withColumnRenamed("Saves","saves").withColumnRenamed("Comments","comments").withColumnRenamed("Shares","shares").withColumnRenamed("Likes","likes").withColumnRenamed("Profile Visits","pv").withColumnRenamed("Follows","follows").withColumnRenamed("Caption","caption").withColumnRenamed("Hashtag1","h1").withColumnRenamed("Hashtag2","h2").withColumnRenamed("Hashtag3","h3")




# In[10]:


insta.columns #shows Columns Name


# In[11]:


type(insta["impressions"]) #its shows the Type of column


# In[12]:


insta=insta.withColumn('impressions',insta["impressions"].cast("int")).withColumn('fhome',insta["fhome"].cast("int")).withColumn('fhashtags',insta["fhashtags"].cast("int")).withColumn('fe',insta["fe"].cast("int")).withColumn('fo',insta['fo'].cast("int")).withColumn('saves',insta['saves'].cast("int")).withColumn('comments',insta['comments'].cast("int")).withColumn('shares',insta['shares'].cast("int")).withColumn('likes',insta['likes'].cast("int")).withColumn('pv',insta['pv'].cast("int")).withColumn('follows',insta['follows'].cast("int"))


# In[13]:


insta.printSchema()


# In[14]:


insta.select("impressions","fhome","h1").show(5) #its use to show multiple columns in single command 


# ##### Step:4 Handling Missing values

# In[15]:


insta.show(5)


# ###### *This Function is shows the how many null value present in the column

# In[16]:


from pyspark.sql.functions import isnan, when, count, col
insta.select([count(when(col(c).isNull(), c)).alias(c) for c in insta.columns]).show()    


# ##### *Using thresholds we can drop the entire row when more than or equal to 3 null values is present in there entire row

# In[17]:


insta.na.drop(thresh=2).show(5)


# ##### *Filling the null values or nan values using impute function

# In[18]:


from pyspark.ml.feature import Imputer


# ###### In our Dataset where null values is present those columns has filled as integer values so we can use only median and mode method 

# In[19]:


insta1=insta.select('fe','comments','shares','follows')


# ###### Impute function for median values

# In[20]:


Imputer1=Imputer(
                 inputCols=insta1.columns,
                 outputCols=['{}.imputed'.format(c) for c in insta1.columns]

).setStrategy('median')


# In[21]:


Imputer1.fit(insta1).transform(insta1).show(5)


# ###### Impute function for mode values

# In[22]:


Imputer2=Imputer(
                 inputCols=insta1.columns,
                 outputCols=['{}.imputed'.format(c) for c in insta1.columns]

).setStrategy('mode')


# In[23]:


Imputer2.fit(insta1).transform(insta1).show(5)


# In[24]:


insta.show()


# In[32]:


i1=insta.na.fill(value='mode').show()


# In[30]:



#Write DataFrame to CSV file
#i1..csv(r"F:\New folder\Instagram_final.csv")

i1.write.mode('overwrite').csv(r"F:\New folder\Instagram_final.csv")


# In[ ]:




