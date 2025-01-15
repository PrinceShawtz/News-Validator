from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess the data
fakeData = pd.read_csv('Fake.csv')
trueData = pd.read_csv('True.csv')

fakeData["class"] = 0
trueData['class'] = 1

fakeDataManualTesting = fakeData.tail(10)
for i in range(23480, 23470, -1):
    fakeData.drop([i], axis=0, inplace=True)

trueDataManualTesting = trueData.tail(10)
for i in range(21416, 21406, -1):
    trueData.drop([i], axis=0, inplace=True)

fakeDataManualTesting['class'] = 0
trueDataManualTesting['class'] = 1

mergeData = pd.concat([fakeData, trueData], axis=0)

data = mergeData.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

def textPreprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(textPreprocessing)

x = data['text']
y = data['class']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xvTrain = vectorization.fit_transform(xTrain)
xvTest = vectorization.transform(xTest)

LR = LogisticRegression()
LR.fit(xvTrain, yTrain)

DT = DecisionTreeClassifier()
DT.fit(xvTrain, yTrain)

RF = RandomForestClassifier(random_state=0)
RF.fit(xvTrain, yTrain)

def labelOutput(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

def detect_news(news):
    news = textPreprocessing(news)
    news_vectorized = vectorization.transform([news])
    
    lr_prediction = LR.predict(news_vectorized)[0]
    dt_prediction = DT.predict(news_vectorized)[0]
    rf_prediction = RF.predict(news_vectorized)[0]
    
    return {
        'LR_prediction': labelOutput(lr_prediction),
        'DT_prediction': labelOutput(dt_prediction),
        'RF_prediction': labelOutput(rf_prediction)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-news', methods=['POST'])
def handle_detect_news():
    news_data = request.json
    news_text = news_data['text']
    
    result = detect_news(news_text)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
