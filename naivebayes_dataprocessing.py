import pandas as pd
import numpy as np
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
import ssl
import heapq
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

training = pd.read_json('Grocery_and_Gourmet_Food_Reviews_training.json', lines=True)
textOnly = training[['overall', 'asin', 'reviewText']]
textOnly.dropna()

per_product = pd.DataFrame({'ProductID': training['asin']})
# drop duplicate rows with the same ProductID
per_product.drop_duplicates(subset=['ProductID'], keep='first', inplace=True)
print(per_product.head())

for value in per_product['ProductID']:
    # print(value)
    # a string with all of the reviews for the same product combined
    reviewList = ""
    # get panda with all of the rows with the same productID
    sameID = training.loc[training['asin'] == value]
    try:
        # combine all of the reviews for the same product into one long string
        per_product.loc[(per_product['ProductID'] == value), "Reviews"] = " ".join(sameID['summary'].tolist())

    except TypeError:
        for review in sameID['summary']:
            reviewList += " " + str(review)
        per_product.loc[(per_product['ProductID'] == value), "Reviews"] = reviewList
        #print(value)
        #print(reviewList)


    review_scores = training.loc[training['asin'] == value]

    # check if product is awesome, create y column of 1's and 0's
    meanScore = review_scores['overall'].mean()
    if meanScore > 4.4:
        product_class = 1
    else:
        product_class = 0
    per_product.loc[(per_product['ProductID'] == value), "Awesome?"] = product_class

pd.option_context('display.max_colwidth', 100)
print(per_product.head(10))

per_product.to_csv('Grocery_ReviewSummary_Training_Data1.csv')