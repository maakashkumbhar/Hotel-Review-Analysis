import numpy as np
import pandas as pd
from sklearn.svm import SVC
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score
from flask import Flask,render_template,request

app =Flask("__main__")



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    
    data = pd.read_csv('reviews.tsv',delimiter='\t',quoting=3)
    corpus = []
    for i in range(0,1000):
        review = re.sub('^a-zA-Z',' ',data['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)



    # Making sparse matrix using a count vextorizer
    cv  = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = data.iloc[:,-1].values


    # Spliting the data into training and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    # Training the Kernel SVM model on the Training set

    classifier_SVC = SVC(kernel = 'linear', random_state = 0)
    classifier_SVC.fit(X_train, y_train)

    # predicting the results as SVC Classifier
    y_pred_SVC = classifier_SVC.predict(X_test)


    #print(np.concatenate((y_pred_SVC.reshape(len(y_pred_SVC),1), y_test.reshape(len(y_test),1)),1))

    # Evaluating the Accuracy
    #cm = confusion_matrix(y_test, y_pred_SVC)
    #print(cm)
    #print(accuracy_score(y_test, y_pred_SVC))
    
    
    new_review = request.form['review']
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier_SVC.predict(new_X_test)
    
    
    if new_y_pred == 1:
        prediction = 'This review is Positive  '
    else:
        prediction = 'This review is Negative'
    
    return render_template('predict.html',prediction_text=prediction)
if __name__ == "__main__":
    app.run(debug=True)