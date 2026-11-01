#!/usr/bin/env python
# coding: utf-8

# ## Exercise 2

# For this exercise , you will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set and very often is a student's first step in Data Analytics! 
# 
# The Dataset has been given to you on D2L. You need to download the .csv file from your assignment folder. The above link is just for a reference story about the data. 
# 
# 
# 1- For this assignment, you need to perform explorotary data analysis and answer at least three hypotheses based on the dataset. You may need to use your knowledge of statiscts to analyze this data.
# 
# Here are three possible hypotheses that you can define for this dataset (you can define your own hypotheses as well):
# 
# - Determine if the survival rate is associated to the class of passenger
# - Determine if the survival rate is associated to the gender
# - Determine the survival rate is associated to the age
# 
# 
# 2- For each hypothesis, you need to make at least one plot. 
# 
# 3- Write a summary of your findings in one page (e.g., summary statistics, plots) and submit the pdf file. Therefore, for part 2 of your assignment, you need to submit one jupyter notebook file and one pdf file.
# 
# This will be your first end to end data analysis project. For this assignment, you will be graded on you overall analysis, and your final report.
# 
# 4- Push your code and project to github and provide the link to your code here.
# 

# https://github.com/Karlacristheljp/Titanic---Exploratory-Data-Analysis-EDA-

# Ensure that your github project is organized to at least couple of main folders, ensure that you have the README file as well:
# 
# Src
# Data
# Docs
# Results
# Read this link for further info: https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510

# In[146]:


# Import galleries
import os
import pandas as pd #library to manipulate data
import matplotlib.pyplot as plt #library for graphics
import seaborn as sns
import numpy as nd
import matplotlib.ticker as mtick
# 2. Configurar el estilo visual de los gráficos
sns.set_theme(style="whitegrid")

# Read the dataset
# Como nuestro notebook está en 'Src', subimos un nivel ('..') y entramos a 'Data'
ruta_titanic = os.path.join("..", "Data", "train.csv") 
#os.path.join("..", "Data", "train.csv") we use this to create a path, I said to python jump out of "Scr" and go in to data and search
#read the document .cvs
df_titanic = pd.read_csv(ruta_titanic)



# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# ## 1. Structure and dimension ##

# #### 1.1 Dataframe dimension and preview

# In[4]:


df_titanic.shape


# In[5]:


df_titanic.head(7)


# #### 1.2 General information and data types

# In[7]:


df_titanic.info()


# **Statistical summary**

# In[31]:


print("Statistical summary")
df_titanic.describe().T[["count", "mean", "std", "min", "50%", "max"]].head(10)


# In[158]:


df_titanic.columns


# **Percentage of Survived**

# In[39]:


df_titanic= df_titanic["Survived"].value_counts(normalize=True) * 100
df_titanic


# ## 2. Missing Values Analysis

# #### 2.1 Quantification of Null Data

# In[42]:


missing_data = df_titanic.isnull().sum()
missing_data = missing_data[ missing_data > 0].sort_values(ascending = False)
print ("Columns with missing values:")
print (missing_data.head(10))


# ## Exploratory Data Analysis (EDA)

# ### 3. Bivariate & Multivariate Analysis (Hypotheses Testing)

# **3.1 Hypothesis #1 Determine if the survival rate is associated to the class of passenger**

# In[118]:


#groupby function groups the passenger first by "Class" and then by "sex" [As a list]
# Then after filtering, I am interested in only the values in ["Survived"] column and calculate the mean()
# .reset_index() for table with a clean structure
df_titanic_by_class= df_titanic.groupby(["Pclass","Sex"])["Survived"].mean()*100
df_titanic_by_class.reset_index()


# In[113]:


df_titanic_by_class= df_titanic.groupby(["Pclass","Sex"])["Survived"].value_counts(normalize=True)*100
df_titanic_by_class.reset_index()


# **Bar graph of Survival rate by Passenger Class**

# In[136]:


#Set up size for the plot
custome_palette= ["green" , "gold", "orange"]

plt.figure(figsize=(9, 5))

ax= sns.barplot(x= "Pclass", y= "Survived", hue="Pclass", data= df_titanic, palette=custome_palette, legend=False)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.title("Survival percentages by Passenger Class", fontsize= 12, fontweight='bold')
plt.xlabel("Passenger Class", fontsize= 10)
plt.ylabel("Survival Rate (Percentage)", fontsize= 10)

# Save the generated plot into the Results folder 
plt.savefig('../Results/hypothesis_class_survival_percentage.png', bbox_inches='tight')
plt.show()


# **Bar graph of Survival rate by Passenger Class and Sex**

# In[152]:


#Set up size for the plot
custome_palette= ["deeppink" , "steelblue"]
plt.figure(figsize=(9, 5))


ax=sns.barplot(x= "Pclass", y= "Survived", hue= "Sex", data= df_titanic, palette= custome_palette)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.title("Survival rate by Passenger Class and Sex", fontsize= 12, fontweight='bold')
plt.xlabel("Passenger Class", fontsize= 10)
plt.ylabel("Survival Rate (%)", fontsize= 10)

# Save the generated plot into the Results folder 
plt.savefig('../Results/hypothesis_class_survival_and sex.png', bbox_inches='tight')
plt.show()


# **Percentage of survival rates**

# In[119]:


df_titanic_by_class= df_titanic.groupby("Pclass")["Survived"].mean()*100
df_titanic_by_class.reset_index()


# **3.2 Hypothesis #2 - Determine if the survival rate is associated to the gender**

# In[52]:


df_titanic_by_class= df_titanic.groupby("Sex")["Survived"].mean().reset_index()
df_titanic_by_class


# **Bar plot of Survival rate by Sex**

# In[151]:


#Set up size for the plot
custome_palette= ["darkcyan" , "rebeccapurple"]
plt.figure(figsize=(9, 5))


ax= sns.barplot(x= "Sex", y= "Survived", hue="Sex", legend=False, data= df_titanic, palette= custome_palette)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.title("Survival rate by Gender", fontsize= 12, fontweight='bold')
plt.xlabel("Gender", fontsize= 10)
plt.ylabel("Survival Rate (%)", fontsize= 10)

# Save the generated plot into the Results folder 
plt.savefig('../Results/hypothesis_survival_rates_by_gender.png', bbox_inches='tight')
plt.show()


# **3.3 Hypothesis #3 Determine the survival rate is associated to the age**

# Mean of group by Age

# In[53]:


df_titanic_by_class= df_titanic.groupby("Age")["Survived"].mean().reset_index()
df_titanic_by_class


# **Histograms of Age Distribution Density by Survival Status**

# In[156]:


custome_palette= ["orangered","limegreen"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))


sns.histplot(df_titanic['Age'], kde=True, color='mediumvioletred', bins=80, ax=axes[0])
axes[0].set_title("Age Distribution Density of All Passengers", fontsize=12)
axes[0].set_xlabel('Age (years)', fontsize=10)
axes[0].set_ylabel('Density', fontsize=10)


sns.kdeplot(data= df_titanic, x='Age', hue="Survived", fill=True, palette=custome_palette ,alpha=0.4, ax=axes[1])
axes[1].set_title("Age Distribution Density by Survival Status", fontsize=12)
axes[1].set_xlabel('Age (years)', fontsize=10)
axes[1].set_ylabel('Density', fontsize=10)

plt.legend(title='Survived', labels=['Yes (1)', 'No (0)'], loc='upper right')
plt.grid(axis='both', linestyle='--', alpha=0.7)

plt.savefig('../Results/Age_Distribution_Density.png', bbox_inches='tight')
plt.show()


# **Histogram of Age and Survival Count (with KDE)**

# In[102]:


custome_palette= ["grey","lightseagreen"]
plt.figure(figsize=(8, 4))
sns.histplot(
    data=df_titanic, 
    x='Age', 
    hue='Survived', 
    multiple='stack', 
    palette=custome_palette, 
    kde=True, 
)
plt.title("Age and Survival Count (with KDE)", fontsize=12, fontweight='bold')
plt.xlabel('Age (years)', fontsize=10)
plt.ylabel('Count', fontsize=10)

plt.legend(title='Survived', labels=['Yes (1)', 'No (0)'], loc='upper right')
plt.grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../Results/hypothesis_age_survival.png', bbox_inches='tight')
plt.show()


# **3.4 Correlation Matrix of Titanic Numerical Variables**

# In[95]:


# Select only the numerical columns 
numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df_titanic[numerical_cols].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5, 
    vmin=-1, 
    vmax=1
)

plt.title('Correlation Matrix of Titanic Numerical Variables', fontsize=12, fontweight='bold')

plt.savefig('../Results/correlation_matrix.png', bbox_inches='tight')

plt.show()

