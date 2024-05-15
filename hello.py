import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

data=pd.read_csv('train.csv')
data.head()

data.shape

x=data['text']
y=data['sentiment']

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

cleaned_data = []
Y = []
for i in range(len(x)):
    if isinstance(x.iloc[i], str) and not pd.isna(y[i]):
        data = re.sub('[^a-zA-Z]', ' ', x.iloc[i])
        data = data.lower().split()
        Y.append(y[i])
        data = [lemmatizer.lemmatize(word) for word in data]
        data = ' '.join(data)
        cleaned_data.append(data)

sentiment_ordering = ['negative', 'neutral', 'positive']

yy = pd.DataFrame(Y)

cv=CountVectorizer(max_features=3000,stop_words=['unit'])
X_fin=cv.fit_transform(cleaned_data).toarray()
X_fin.shape
X_fin

y_reshaped = np.ravel(Y)

model=MultinomialNB()

X_train,X_test,y_train,y_test=train_test_split(X_fin,y_reshaped,test_size=0.3)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

def classify_tweet(tweet):
  cleaned_tweet = re.sub('[^a-zA-Z]', ' ', tweet)
  cleaned_tweet = cleaned_tweet.lower().split()
  cleaned_tweet = [lemmatizer.lemmatize(word) for word in cleaned_tweet]
  cleaned_tweet = ' '.join(cleaned_tweet)

  X_new = cv.transform([cleaned_tweet]).toarray()

  prediction = model.predict(X_new)
  if prediction[0]=='0':
    return "Negative"
  elif prediction[0]=='1':
    return "Neutral"
  else:
    return "Positive"


while True:
  tweet = input("Enter a tweet: ")
  if tweet == '':
    break

  sentiment = classify_tweet(tweet)
  print(f"Sentiment: {sentiment}")
