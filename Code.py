

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import names
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Function to convert a string to binary
def strToBinary(s):
    binary_result = ''.join(format(ord(char), '08b') for char in s)
    return binary_result

# Function to convert binary to decimal
def binaryToDecimal(binary):
    decimal_result = int(binary, 2)
    return decimal_result

# Read data and preprocess
genuine_users = pd.read_csv("data/users.csv")
fake_users = pd.read_csv("data/fusers.csv")
x = pd.concat([genuine_users, fake_users])
X = pd.DataFrame(x)
t = len(fake_users) * ['Genuine'] + len(genuine_users) * ['Fake']
er = pd.Series(t)
X['label'] = pd.DataFrame(er)
label = {'Genuine': 0, 'Fake': 1}
X.label = [label[item] for item in X.label]
y = X['label'].values

# Gender classification using NLTK
def gender_features(word):
    return {'last_letter': word[-1]}

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

a = []
for i in X['name']:
    vf = classifier.classify(gender_features(i))
    a.append(vf)
X['gender'] = pd.DataFrame(a)
lang_list = list(enumerate(np.unique(X['lang'])))
lang_dict = {name: i for i, name in lang_list}
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['lang_code'] = X['lang'].map(lambda x: lang_dict[x]).astype(int)
feature_columns_to_use = ['gender', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang_code']
ty = X[feature_columns_to_use].values

# Fit the KNeighborsClassifier with feature array 'ty' outside of the route
knn = KNeighborsClassifier()
knn.fit(ty, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    a = request.form['name']
    usname = classifier.classify(gender_features(a))
    print("Predicted Gender:", usname)

    if usname == "male":
        gender = 0
    else:
        gender = 1

    statuses_count = int(request.form['statuses_count'])
    followers_count = int(request.form['followers_count'])
    friends_count = int(request.form['friends_count'])
    favourites_count = int(request.form['favourites_count'])
    listed_count = int(request.form['listed_count'])
    lang_code = int(request.form['lang_code'])

    new = {"gender": gender, "statuses_count": statuses_count, "followers_count": followers_count,
           "friends_count": friends_count, "favourites_count": favourites_count, "listed_count": listed_count,
           "lang_code": lang_code}
    de = pd.DataFrame(new, index=[0])
    de.to_csv("new.csv")
    re = pd.read_csv("new.csv")
    rs = re.iloc[:, 1:].values
    rs_reshaped = rs.reshape(1, -1)

    print("Feature Values:", rs_reshaped)

    try:
        prediction = knn.predict(rs_reshaped)
        print("Raw Prediction:", prediction)
        result = "Genuine" if prediction == 0 else "Fake"
        print("Predicted Label:", result)
    except Exception as e:
        result = "Error during prediction: {}".format(e)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)



