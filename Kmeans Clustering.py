# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 06:29:42 2018

@author: lenovo
"""

   We will use a data frame with 777 observations on the following 18 variables.
•	Private A factor with levels No and Yes indicating private or public university
•	Apps Number of applications received
•	Accept Number of applications accepted
•	Enroll Number of new students enrolled
•	Top10perc Pct. new students from top 10% of H.S. class
•	Top25perc Pct. new students from top 25% of H.S. class
•	F.Undergrad Number of fulltime undergraduates
•	P.Undergrad Number of parttime undergraduates
•	Outstate Out-of-state tuition
•	Room.Board Room and board costs
•	Books Estimated book costs
•	Personal Estimated personal spending
•	PhD Pct. of faculty with Ph.D.’s
•	Terminal Pct. of faculty with terminal degree
•	S.F.Ratio Student/faculty ratio
•	perc.alumni Pct. alumni who donate
•	Expend Instructional expenditure per student
•	Grad.Rate Graduation rate
##Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## Get the Data - ** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.
df = pd.read_csv("C:\\Users\\ravi\\Desktop\\INDRAS ACADEMY\\phyton programming\\College.csv",index_col=0)


##Check the head of the data
df.head()

df.info()

df.describe()


##data visualizations

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

##Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

##Create a stacked histogram showing Out of State Tuition based on the Private column
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

###Create a similar histogram for the Grad.Rate column
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
## error graduate rate more than 100%. Need to adjust to 100%

df[df['Grad.Rate'] > 40][df['Private']=='No']
#df[(df.'Grad.Rate' > 40)&(df'Private'=='No')]
df[df['Grad.Rate']> 100]

df['Grad.Rate']['Cazenovia College'] = 100


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

## K Means Cluster Creation

from sklearn.cluster import KMeans
###K Means model with 2 clusters
kmeans = KMeans(n_clusters=2)

## fit the data for all the variables except private labeles
kmeans.fit(df.drop('Private',axis=1))

### To check centralized values

kmeans.cluster_centers_



## Evaluation 
##Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)
df.head(50)
df
## To check labels
kmeans.labels_


from sklearn.metrics import classification_report,confusion_matrix
##print(confusion_matrix(df['Cluster'],kmeans.labels_]))

print(classification_report(df['Cluster'],kmeans.labels_))


 