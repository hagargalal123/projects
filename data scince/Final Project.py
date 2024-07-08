#!/usr/bin/env python
# coding: utf-8

# # Question
# 1-what is the most expensive laptop on store?
# 
# 2-what is the average Price of Manufacturer?
# 
# 3-What's the 5 top highest storage for Apple?
# 
# 4-Which model that has the highest screen size ?
# 
# 5-is there's a correlation between RAM and Price ?
# 
# 6-what is the 10 lowest price?
# 
# 7-how many laptops that have weight > 4.4kg ?
# 
# 8- what is the first 10 laptops that in Gaming Category and avg of its Price?
# 
# 9-Which model that his Hyprid not equal 0 ?
# 
# 10-how many laptops that have price > 16.9k ?
# 
# 11-What is the relationship between laptop screen size and weight?
# 
# 12-What is the average weight of laptops in the dataset?
# 
# 13-how many laptops that have 13.5" screen size ?
# 
# 14-What is the average price of laptops in the dataset?
# 
# 15-How many laptop for each brand?
# 
# 16-What is the most common brand of laptops in the dataset?
# 
# 17-Is there any correlations between the laptop specifications (CPU, RAM, storage) and price?
# 
# 18-Are there any outliers in the dataset with unusually high or low prices? If so, what might explain these outliers?
# 
# 19-What are the least common operating systems used by the laptops in the dataset and how many times it found?
# 
# 20-Are there any noticeable differences in price or specifications between laptops with different operating systems?
# 
# 21-how many laptops that have specifications with (Model = HP && GPU = AMD Radeon 530) ?
# 
# 22- How many models of laptops in dataset?
# 
# 23-what are laptops that have No OS in their operating ?
# 
# 24-what is the lap that has the high specifications ?
# 
# 25-How many model names laptops in the dataset ?
# 
# 26-what is the 5 highest price?

# # Import Library

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # load the Data

# In[2]:


df=pd.read_csv("desktop/laptop.csv")


# In[3]:


df


# # Data exploration

# In[4]:


df.info()


# # Data Cleaning

# # missing value

# In[5]:


df.isna().sum()


# In[6]:


from sklearn.impute import SimpleImputer


# In[7]:


imputer = SimpleImputer(strategy= 'most_frequent')
df['Operating System Version']=imputer.fit_transform(df[['Operating System Version']])


# In[8]:


df.isna().sum()


# # check if there is a dublicated value

# In[9]:


df.duplicated().sum()


# In[10]:


df.info()


# # fix RAM

# In[11]:


for RAM in df['RAM'].unique():
    print(RAM)


# In[12]:


df["RAM"] = df["RAM"].str.replace("GB", "")
df["RAM"] = df["RAM"].astype(float)


# In[14]:


df


# In[15]:


df.info()


# # fix Weight

# In[16]:


for Weight in df['Weight'].unique():
    print(Weight)


# In[17]:


df["Weight"] = df["Weight"].str.replace("kg", "")
df["Weight"] = df["Weight"].astype(float)


# In[18]:


df


# In[19]:


df.info()


# # fix Screen Size

# In[20]:


for Screen_Size in df['Screen Size'].unique():
    print(Screen_Size)


# In[21]:


df["Screen Size"] = df["Screen Size"].str.replace('"', "")
df["Screen Size"] = df["Screen Size"].astype(float)


# In[22]:


df


# In[23]:


df.info()


# # fix Storage

# In[24]:


for Storage in df[' Storage'].unique():
    print(Storage)


# In[25]:


df[' Storage'] = df[' Storage'].astype(str).replace('\.0','',regex=True)
df[' Storage'] = df[' Storage'].astype(str).replace('GB','')
df[' Storage'] = df[' Storage'].astype(str).replace('TB','000')
new = df[' Storage'].str.split('+',n=1,expand=True)

df['first'] = new[0]
df['first'] = df['first'].str.strip()
df['sec'] = new[1]


# In[26]:


df['l1hdd'] = df['first'].apply(lambda x: 1 if "HDD" in x else 0)
df['l1ssd'] = df['first'].apply(lambda x: 1 if "SSD" in x else 0)
df['l1hybrid'] = df['first'].apply(lambda x: 1 if "Hybrid" in x else 0)
df['l1fs'] = df['first'].apply(lambda x: 1 if "Flash Storage" in x else 0)
df['first'] = df['first'].str.replace(r'\D','')
df['sec'].fillna('0',inplace=True)


# In[27]:


df['l2hdd'] = df['sec'].apply(lambda x: 1 if "HDD" in x else 0)
df['l2ssd'] = df['sec'].apply(lambda x: 1 if "SSD" in x else 0)
df['l2hybrid'] = df['sec'].apply(lambda x: 1 if "Hybrid" in x else 0)
df['l2fs'] = df['sec'].apply(lambda x: 1 if "Flash Storage" in x else 0)
df['sec'] = df['sec'].str.replace(r'\D','')


# In[28]:


df['first'] = df['first'].astype('int')
df['sec'] = df['sec'].astype('int')


# In[29]:


df['HDD'] = df['first']*df['l1hdd'] + df['sec']*df['l2hdd']
df['SSD'] = df['first']*df['l1ssd'] + df['sec']*df['l2ssd']
df['Hybrid'] = df['first']*df['l1hybrid'] + df['sec']*df['l2hybrid']
df['FS'] = df['first']*df['l1fs'] + df['sec']*df['l2fs']

df.drop(columns=['first','sec','l1hdd','l1ssd','l1hybrid','l1fs','l2hdd','l2ssd','l2fs','l2hybrid'],inplace=True)


# In[30]:


df=df.drop(' Storage',axis=1)


# In[31]:


df


# In[32]:


df.info()


# In[33]:


for Storage in df['Price'].unique():
    print(Storage)


# In[34]:


df['Price_ln']=df['Price'].apply(np.log)
df=df.drop('Price',axis=1)
df


# In[35]:


for review in df['Manufacturer'].unique():
    print(review)


# In[36]:


for review in df['Model Name'].unique():
    print(review)


# In[37]:


for review in df['Category'].unique():
    print(review)


# In[38]:


for review in df['Screen'].unique():
    print(review)


# In[39]:


for review in df['CPU'].unique():
    print(review)


# In[40]:


for review in df['GPU'].unique():
    print(review)


# In[41]:


for review in df['Operating System'].unique():
    print(review)


# In[42]:


for review in df['Operating System Version'].unique():
    print(review)


# In[43]:


df.describe()


# # outlier

# In[44]:


sns.boxplot(x='Price_ln',data=df)


# In[45]:


sns.boxplot(x='RAM',data=df)


# In[46]:


sns.boxplot(x='Screen Size',data=df)


# In[47]:


sns.boxplot(x='Weight',data=df)


# In[48]:


df


# # Question 

# # 1-what is the most expensive laptop on store?

# In[49]:


df[df['Price_ln']==df['Price_ln'].max()]


# # 2- what is the average Price of Manufacturer?

# In[50]:


df.groupby('Manufacturer').mean()['Price_ln']


# # 3-What's the 5 top highest Storage ('SSD') for Apple?

# In[51]:


df[df['Manufacturer']=='Apple'].sort_values(by='SSD',ascending=False).head(5)


# # 4-Which model that has the highest screen size ?

# In[52]:


df[df['Screen Size']==df['Screen Size'].max()]


# In[53]:


sns.boxplot(x='Screen Size',data=df)


# # 5-is there's a correlation between RAM and Price ?

# In[54]:


col1=df['RAM']

col2=df['Price_ln']

corr=col1.corr(col2)

print("corrleation between RAM and Price is:",corr)


# In[55]:


sns.heatmap(df.corr(),annot=True)


# # 6-what is the 10 lowest price?

# In[56]:


df.nsmallest(10, 'Price_ln')


# # 7-how many laptops that have weight > 4.4kg ?

# In[57]:


df[df['Weight']>4.4]


# # 8- what is the first 10 laptops that in Gaming Category and avg of its Price?

# In[59]:


gaming_model = df[df['Category'] == 'Gaming'].head(10)

avg_price = gaming_model['Price_ln'].mean()

print("Gaming models and their average price:")

print(gaming_model[['Manufacturer','Category', 'Price_ln']])

print("Average price of gaming models:", avg_price)


# # 9-Which model that his Hyprid not equal 0 ?

# In[60]:


diff_model = df[df['Hybrid'] != 0]

print(diff_model[['Manufacturer','Model Name','Hybrid']])


# # 10-how many laptops that have price > 16.9k ?

# In[62]:


df[df['Price_ln']>16.9]


# # 11-What is the relationship between laptop screen size and weight?

# In[63]:


col1=df['Weight']

col2=df['Screen Size']

corr=col1.corr(col2)

print("corrleation between Manufacturer and Screen Size is:",corr)


# In[64]:


sns.heatmap(df.corr(),annot=True)


#  # 12-What is the average weight of laptops in the dataset?
# 

# In[65]:


df.groupby('Manufacturer').mean()['Weight']


# # 13-how many laptops that have 13.5" screen size ?
# 

# In[66]:


Model = df[df['Screen Size'] == 13.5]

print(Model[['Manufacturer','Screen Size']])


# # 14-What is the average price of laptops in the dataset?

# In[68]:


AVG = df["Price_ln"].mean()

print('the AVG price of ALL_laptops is :',AVG)


# # 15-How many laptop for each brand? 

# In[69]:


brand_counts = df['Manufacturer'].value_counts()

print(brand_counts)


# # 16-What is the most common brand of laptops in the dataset? 

# In[70]:


df['Manufacturer'].value_counts().nlargest(1)


# In[71]:


sns.set(rc={'figure.figsize':[10,10]},font_scale=1.2)

sns.countplot(y='Manufacturer',data=df)


# # 17-Is there any correlations between the laptop specifications (CPU, RAM, storage) and price?

# In[72]:


df.corr()


# In[73]:


df.corr()['Price_ln']


# In[74]:


sns.heatmap(df.corr(),annot=True)


# # 18-Are there any outliers in the dataset with unusually high or low prices? If so, what might explain these outliers?
# 
# we don't have any outliers 
# 

# In[75]:


z_scores = np.abs((df['Price_ln'] - df['Price_ln'].mean()) / df['Price_ln'].std())

outliers = df[z_scores > 3]

print(outliers)


# # 19-What are the least common operating systems used by the laptops in the dataset and how many times it found?

# In[76]:


df['Operating System Version'].value_counts().nsmallest(1)


# # 20-Are there any noticeable differences in price or specifications between laptops with different operating systems?

# In[77]:


# group the laptops by operating system and calculate summary statistics
os_stats = df.groupby('Operating System Version').agg({'Price_ln': ['count', 'mean', 'std'], 'RAM': ['mean', 'std']})

# print the summary statistics for each operating system
print(os_stats)


# # 21-how many laptops that have specifications with (Model = HP && GPU = AMD Radeon 530) ?

# In[78]:


filtered_df = df[(df['Manufacturer'] == 'HP') & (df['GPU'] == 'AMD Radeon 530')]

print(filtered_df[['Manufacturer', 'GPU']])


# # 22- How many models of laptops in dataset?

# In[80]:


df['Manufacturer'].nunique()


# # 23-what are laptops that have No OS in their operating ?

# In[81]:


filtered_df = df[(df['Operating System'] == 'No OS')]

print(filtered_df[['Manufacturer', 'Operating System']])


# In[82]:


sns.countplot(x='Operating System',data=df)


# # 24-what is the lap that has the high specifications ?

# In[83]:


sorted_df = df.sort_values(by=['RAM', 'Weight'], ascending=[False, True])

best_laptop = sorted_df.iloc[0]

print("The laptop with the highest specifications is:")
print(best_laptop)


# # 25-How many model names laptops in the dataset ?

# In[84]:


df['Model Name'].nunique()


# # 26-what is the 5 highest price?

# In[85]:


df.nlargest(5, 'Price_ln')



# In[87]:


df1 = df.nlargest(5, 'Price_ln')
plt.boxplot(df1['Price_ln'])
plt.title('Boxplot of Prices for Top 5 Laptops')
plt.ylabel('Price')
plt.show()


# In[88]:


df


# # Machine Learning

# In[95]:


df.to_csv(r"C:\Users\Qaiaty store\Desktop\laptop.csv", index=False)


# In[96]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[98]:


data = pd.read_csv("C:/Users/Qaiaty store/Desktop/laptop.csv")


# In[99]:


selected_features = ['Screen Size', 'RAM', 'CPU', 'GPU', 'Weight']
X = data[selected_features]
y = data['Price_ln']


# In[100]:


transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['GPU', 'CPU'])
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# In[101]:


print('Train Set Score:', model.score(X_train, y_train))
print('Test Set Score:', model.score(X_test, y_test))


# In[ ]:




