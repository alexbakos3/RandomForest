import pandas as pd
import numpy as np

training = pd.read_csv('Grocery_ReviewText_Training_Data3.csv')

print(training.columns)
del training['Unnamed: 0']
print(training.head())
X_text = training['Reviews']
Y = training['Awesome?']

print(X_text.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=4000, min_df=3, max_df=0.8, stop_words='english')
processed_features = vectorizer.fit_transform(X_text.to_list())
processed_features = processed_features.toarray()

# print(processed_features)
print(processed_features.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, Y, test_size=0.33)

# summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(["I Like to eat"])
# summarize encoded vector
# print(vectorizer.get_stop_words())
print(vector.shape)
print(vector.toarray())
# print(vectorizer.vocabulary_)

from sklearn.ensemble import RandomForestRegressor
#
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
y_pred = np.round(y_pred)
print(confusion_matrix(y_test.tolist(),y_pred))

print(classification_report(Y,y_pred))
print(accuracy_score(Y, y_pred))