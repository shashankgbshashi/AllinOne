import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = [('shashank','shashi'),('roshan','rosh')]
#dataset = pd.DataFrame(a)
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for j in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][j])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.05,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

