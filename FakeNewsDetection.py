import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

fakeData=pd.read_csv("C:/Users/Prince Shawtz/Documents/newsdetection/Fake.csv")
trueData=pd.read_csv("C:/Users/Prince Shawtz/Documents/newsdetection/True.csv")

fakeData["class"]=0
trueData['class']=1

fakeDataManualTesting = fakeData.tail(10)
for i in range(23480, 23470, -1):
    fakeData.drop([i], axis=0, inplace=True)

trueDataManualTesting = trueData.tail(10)
for i in range(21416, 21406, -1):
    trueData.drop([i], axis=0, inplace=True)

fakeDataManualTesting['class']=0
trueDataManualTesting['class']=1

mergeData=pd.concat([fakeData, trueData], axis = 0)

data=mergeData.drop(['title', 'subject', 'date'], axis = 1)

data = data.sample(frac = 1)

data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)

def textPreprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

data['text'] = data['text'].apply(textPreprocessing)

x = data['text']
y = data['class']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xvTrain = vectorization.fit_transform(xTrain)
xvTest = vectorization.transform(xTest)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
print(LR.fit(xvTrain, yTrain))

logisticRegressionPrediction = LR.predict(xvTest)
print(LR.score(xvTest, yTest))

print(classification_report(yTest, logisticRegressionPrediction))

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
print(DT.fit(xvTrain, yTrain))

decisionTreePrediction = DT.predict(xvTest)
print(DT.score(xvTest, yTest))

print(classification_report(yTest, logisticRegressionPrediction))

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
print(RF.fit(xvTrain, yTrain))

randomForestPrediction = RF.predict(xvTest)
print(RF.score(xvTest, yTest))
print (classification_report(yTest, randomForestPrediction))


def labelOutput(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"


def manualTesting(news):
    newsTest = {"text": [news]}
    newDefTest = pd.DataFrame(newsTest)
    newDefTest['text'] = newDefTest["text"].apply(textPreprocessing)
    newXTest = newDefTest["text"]
    newXVTest = vectorization.transform(newXTest)
    lrPrediction = LR.predict(newXVTest)
    dtPrediction = DT.predict(newXVTest)
    rfPrediction = RF.predict(newXVTest)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction:{}".format(
        labelOutput(lrPrediction[0]),
        labelOutput(dtPrediction[0]),
        labelOutput(rfPrediction[0])))

print("Please enter news you would like to check: ")
news = str(input())
manualTesting(news)