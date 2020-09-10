# -*- coding: utf-8 -*-
import numpy as np
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


model = pickle.load(open('model.pkl','rb'))

review = 'i hate this resturant'
review = re.sub('^a-zA-Z',' ',review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
review = ' '.join(review)
new_corpus = [review]
cv  = CountVectorizer()
X_new = cv.fit_transform(new_corpus).toarray()
final_feature = [np.array(X_new)]
y_pred_new = model.predict(final_feature)

print(y_pred_new)