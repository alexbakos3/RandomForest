import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
training = pd.read_csv('Grocery_ReviewSummary_Training_Data1.csv')

print(training.columns)
del training['Unnamed: 0']

X_text = training['Reviews']
Y = training['Awesome?']


from sklearn.feature_extraction.text import TfidfVectorizer
# create bag of words TF-IDF vectorizer,
bow_vectorizer = TfidfVectorizer (max_features=4000, min_df=3, max_df=0.8, stop_words='english')
# process each product
processed_features = bow_vectorizer.fit_transform(X_text.to_list())
processed_features = processed_features.toarray()


# split X and y into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, Y, test_size=0.33)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test.tolist(),y_predictions))
print(classification_report(y_test.tolist(),y_predictions))
print(accuracy_score(y_test.tolist(), y_predictions))

print(y_predictions)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

print(confusion_matrix(y_test.tolist(),y_pred))
print(classification_report(y_test.tolist(),y_pred))
print(accuracy_score(y_test.tolist(), y_pred))