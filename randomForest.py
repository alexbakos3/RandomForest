import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

training = pd.read_json('Grocery_and_Gourmet_Food_Reviews_training.json', lines=True, nrows= 500000)

print(list(training.columns))

processedReviews = pd.DataFrame({'ProductID': training['asin']})
processedReviews.drop_duplicates(subset=['ProductID'], keep='first', inplace=True)
print(processedReviews.head())

for value in processedReviews['ProductID']:
    # print(value)
    reviewList = ""
    sameID = training.loc[training['asin'] == value]
    try:
        processedReviews.loc[(processedReviews['ProductID'] == value), "Reviews"] = " ".join(sameID['reviewText'].tolist())
    except TypeError:
        for review in sameID['reviewText']:
            reviewList += " " + str(review)
        processedReviews.loc[(processedReviews['ProductID'] == value), "Reviews"] = reviewList
        #print(value)
        #print(reviewList)
    y_scores = training.loc[training['asin'] == value]
    meanScore = y_scores['overall'].mean()
    #print(meanScore)
    if meanScore > 4.4:
        product_class = 1
    else:
        product_class = 0
    processedReviews.loc[(processedReviews['ProductID'] == value), "Awesome?"] = product_class

pd.option_context('display.max_colwidth', 100)
print(processedReviews.head(10))

processedReviews.to_csv('Grocery_ReviewText_Training_Data3.csv')


