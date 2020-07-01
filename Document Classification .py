#!/usrimport nltk  
import numpy as np  
import random  
import string
import bs4 as bs  
import urllib.request  
import re 
import pandas as pd
import random
from scipy.sparse import coo_matrix
from sklearn.utils import resample
import bs4 as bs 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import imblearn
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

#Import the labeled files
pdfiles=pd.read_excel(r'************************************************',sheet_name=False)
pdfiles=pd.DataFrame(pdfiles)
pdfiles.head()

#Name the x and y axis of the plot and visualize the barplot of the classes
ax = pdfiles['Quality'].value_counts(sort=False).plot(kind='barh')
ax.set_xlabel('Number of Samples in training Set')
ax.set_ylabel('Label')


# example of random oversampling to balance the class distribution
# define dataset
# summarize class distribution
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(pdfiles['File'].values.reshape(-1,1),pdfiles['Quality'])
# summarize class distribution
print(Counter(y_over))

#Visualize the classes after the oversampling
ax = y_over.value_counts(sort=False).plot(kind='barh')
ax.set_xlabel('Number of Samples in training Set')
ax.set_ylabel('Label')

#Create a new dataframe with oversampled data
oversampled=pd.DataFrame(X_over,y_over)
oversampled.reset_index(inplace=True)
oversampled.columns=['Quality','Files']
oversampled

#Vectorize the data or convert the text into numeric values to perform classification (term frequence - inverse document frequency)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(oversampled.Files).toarray()
labels = oversampled.Quality
features.shape
df=[features,labels]
df=pd.DataFrame(df)
print(df[[0]])

#Split the dataset into trianing and test set
X_train, X_test, y_train, y_test = train_test_split(oversampled['Files'], oversampled['Quality'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


#Fit Four different types of models and use 15 fold cross validation
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 15
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    print(accuracies)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

#visualize the box plot of prediction accuracy for each model
plt.figure(figsize=(13,7))
sns.boxplot(x='model_name', y='accuracy', data=cv_df).set_title('Models Prediction Accuracy',fontsize=30)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=10, jitter=True, edgecolor="gray", linewidth=2)
cv_df.groupby('model_name').accuracy.mean()


#Predict the classes using Linear Support Vector Classifier
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, oversampled.index, test_size=0.30, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




