#!/usr/bin/env python
# coding: utf-8

# **Loan Qualification System**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("train.csv")


# In[3]:


data.head()


# **Exploring the Data**
# 

# In[4]:


data.info() # size: 614 rows and 13 columns; some data missing


# In[5]:


data.describe()


# In[6]:


pd.crosstab(data['Credit_History'], data['Loan_Status'], margins=True)


# In[7]:


data['ApplicantIncome'].hist(bins=20) # Skewed histogram so need to normalize data
data['CoapplicantIncome'].hist(bins=20)


# In[8]:


data['LoanAmount_log']=np.log(data['LoanAmount'])
data['LoanAmount_log'].hist(bins=20)


# **Data Cleaning and Filling in Missing Data**

# In[9]:


data.apply(lambda x: sum(x.isnull()),axis=0) # Checking missing values in each column of train dataset


# In[10]:


data['Gender'].fillna(data['Gender'].mode()[0], inplace=True) # Fill in the Mode for a Categorical variable


# In[11]:


data['Married'].fillna(data['Married'].mode()[0], inplace=True)


# In[12]:


data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)


# In[13]:


data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)


# In[14]:


data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())# Fill in the mean for a numerical value
data.LoanAmount_log = data.LoanAmount_log.fillna(data.LoanAmount_log.mean())


# In[15]:


data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)


# In[16]:


data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)


# In[17]:


data.apply(lambda x: sum(x.isnull()),axis=0) 


# **Normalizing Incomes**

# In[18]:


data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['TotalIncome_log']=np.log(data['TotalIncome'])
data['TotalIncome_log'].hist(bins=20)


# **Splitting Data into Independent and Dependent datasets**

# In[19]:



X = data.iloc[:, 1: 12].values
Y = data.iloc[:, 12].values


# In[20]:


X


# In[21]:


Y


# **Spliting Data into Training and Test sets**

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# In[23]:


X_train


# **Encoding Categorical Data into Numerical Data for Test and Train Datasets**

# In[24]:


# Encoding categorical data in independent variable into numerical data
from sklearn.preprocessing import LabelEncoder # New instance of labelencoder
labelencoder_X = LabelEncoder()

for i in range(0, 5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])

X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])


# In[25]:


X_train


# In[26]:


# Encoding dependent variable (approve/reject) into numerical data
labelencoder_Y = LabelEncoder() 
Y_train = labelencoder_Y.fit_transform(Y_train)


# In[27]:


Y_train


# In[28]:


# Encoding categorical data into numerical data for test attributes
for i in range(0, 5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])
    
X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])


# In[29]:


X_test


# In[30]:


labelencoder_Y = LabelEncoder() 
Y_test = labelencoder_Y.fit_transform(Y_test)


# In[31]:


Y_test


# In[32]:


#Scaling data for analysis
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# **Applying Principal Component Analysis**

# In[33]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_


# **Logistic Regression**

# In[34]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[35]:


# Predicting the Test set results
y=Y_pred = classifier.predict(X_test)


# In[36]:


Y_pred


# In[37]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(Y_pred, Y_test))


# In[38]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print('True Negatives: 145, False Positives: 60')


# In[39]:


# Visualising the training dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[40]:


# Visualising the test dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# **Decision Tree Classification**

# In[41]:


# Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[42]:


# Predicting the test dataset results
Y_pred = classifier.predict(X_test)
Y_pred


# In[43]:


# Measuring Accuracy of the prediction
from sklearn import metrics
print('The accuracy of Decision Tree Classifier is: ', metrics.accuracy_score(Y_pred, Y_test))


# In[44]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print('False Negatives: 37, False Positives: 28')


# In[45]:


# Visualising the training dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


# In[46]:


# Visualising the test dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# **Naive Bayes**

# In[47]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[48]:


# Predicting the test dataset results
Y_pred = classifier.predict(X_test)
Y_pred


# In[49]:


# Measuring Accuracy of the prediction
from sklearn import metrics
print('The accuracy of Decision Tree Classifier is: ', metrics.accuracy_score(Y_pred, Y_test))


# In[50]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print('False Negatives: 2, False Positives: 57')


# In[51]:


# Visualising the training dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[52]:


# Visualising the test dataset results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

