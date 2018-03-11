import math
from collections import OrderedDict
import pandas as pd
from nltk import FreqDist
from nltk.util import ngrams
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.helper import read_data, preprocess_tweet, read_words, create_base_features, undersample, plotPrecRecCurve, \
    plotTrainTestLines, precision_recall_values


# Create columns of bigrams
def create_bigram_columns(bigrams):
    cols = list()
    for bigr in bigrams:
        word1, word2 = bigr
        bigr = "_".join((word1, word2))
        cols.append(bigr)

    return cols


# Assign 0/1 to bigram columns for each row
def count_bigrams_boolean(df_row):
    tokens = df_row["tokenized_text"]
    for bigram in ngrams(tokens, 2):
        word1, word2 = bigram
        bigram = "_".join((word1, word2))
        if bigram in df_row.index:
            df_row[bigram] = 1
    return df_row


tweets_path = "D:/tweets.csv"
positive_words_path = 'D:/positive-words.txt'
negative_words_path = 'D:/negative-words.txt'
emoticons_path = "D:/emoticons.txt"

# Read data and preprocess
tweet_data = read_data(tweets_path)
tweet_data["Tweet"] = tweet_data["Tweet"].apply(preprocess_tweet)

# Read positive/negative words
positive_words = read_words(positive_words_path)
negative_words = read_words(negative_words_path)

tweet_data = create_base_features(tweet_data, positive_words, negative_words, True)

# Get all feature columns
base_features = list(tweet_data.columns.values)
base_features.remove('Category')
base_features.remove('Tweet')
# print(base_features)

# Encode labels
le = LabelEncoder()
tweet_data["Category"] = le.fit_transform(tweet_data["Category"])
tweet_data["Category"].head()

train_X, test_X, train_y, test_y = train_test_split(tweet_data[base_features], tweet_data['Category'], test_size=0.1)
print("train size : " + str(len(train_X)))
# print(train_X.head())
print("test size : " + str(test_X.shape[0]))

# Get all bigrams
train_bigrams = {}
for train_tokens in train_X["tokenized_text"]:
    for bigram in ngrams(train_tokens, 2):
        train_bigrams[bigram] = train_bigrams.get(bigram, 0) + 1

print("train bigrams :", len(train_bigrams))

# Keep frequent bigrams
train_bigrams_filtered = {}
for k, v in train_bigrams.items():
    if v > 2:
        train_bigrams_filtered[k] = v

print("train bigrams filtered :", len(train_bigrams_filtered))

# Add bigram columns to train/test dataframes
train_bigram_columns = create_bigram_columns(train_bigrams_filtered)

train_X = pd.concat([train_X, pd.DataFrame(columns=train_bigram_columns)])
train_X.fillna(0, inplace=True)
test_X = pd.concat([test_X, pd.DataFrame(columns=train_bigram_columns)])
test_X.fillna(0, inplace=True)

train_X = train_X.apply(count_bigrams_boolean, axis=1)
test_X = test_X.apply(count_bigrams_boolean, axis=1)

# print(le.inverse_transform(0))  # negative
# print(le.inverse_transform(1))  # neutral
# print(le.inverse_transform(2))  # positive

# Calculate Total Entropy
negative_tweets = len(train_y[train_y == 0])  # negative
neutral_tweets = len(train_y[train_y == 1])  # neutral
positive_tweets = len(train_y[train_y == 2])  # positive
all_tweets = negative_tweets + neutral_tweets + positive_tweets

tweet_probabilities = [negative_tweets / all_tweets, neutral_tweets / all_tweets, positive_tweets / all_tweets]
entropy_per_class = (prob * math.log(prob, 2) for prob in tweet_probabilities)
total_entropy = 0
for entropy in entropy_per_class:
    total_entropy -= entropy

# Calculate Information Gain - for boolean features
IG_dict = {}  # for the final values

features = base_features
features.remove('tokenized_text')
# For each column - except 'tokenized_text'
for col in (train_bigram_columns + features):
    # Convert all values to boolean
    train_X[col] = train_X[col] > 0
    # Add columns and labels in one dataframe
    temp_df = pd.DataFrame()
    temp_df[col] = train_X[col]
    temp_df["label"] = train_y

    # For attribute 'col' equal to 0
    # Get the respective lines
    df0 = temp_df[temp_df[col] == 0]
    # Count occurences of each class(positive/negative/neutral)
    freqdist0 = FreqDist(df0["label"])
    # Get probability of each class
    probabilities0 = [freqdist0.freq(label) for label in freqdist0]
    # Calculate cross entropy of X=0
    Hc0 = -sum(prob * math.log(prob, 2) for prob in probabilities0)

    # For attribute 'col' equal to 1
    # Get the respective lines
    df1 = temp_df[temp_df[col] == 1]
    # Count occurences of each class(positive/negative/neutral)
    freqdist1 = FreqDist(df1["label"])
    # Get probability of each class
    probabilities1 = [freqdist1.freq(label) for label in freqdist1]
    # Calculate cross entropy of X=1
    Hc1 = -sum(prob * math.log(prob, 2) for prob in probabilities1)

    # Caclulate probabilities for each value of 'col' (0/1)
    freqdist = FreqDist(temp_df[col])
    probabilities = {}
    for label in freqdist:
        probabilities[label] = freqdist.freq(label)

    # Caclulate Information Gain of 'col'
    IG = total_entropy - (probabilities.get(0, 0) * Hc0 + probabilities.get(1, 0) * Hc1)
    IG_dict[col] = IG
    # print(col,IG)

# Sort IGs in reverse order
IG_dict = OrderedDict(sorted(IG_dict.items(), key=lambda x: x[1], reverse=True))

# Find best columns
number_of_columns = 1000
best_columns = list()
for i, col in enumerate(IG_dict):
    if (i < 10):  print(i, col, IG_dict[col])
    best_columns.append(col)
    if i >= number_of_columns - 1:
        break

# Filter the desired columns
train_X = train_X[best_columns]
test_X = test_X[best_columns]

train_X_np, train_y_np = undersample(train_X.values, train_y.values)
print(train_X_np.shape)
print(train_y_np.shape)

#################################################################################################################################################
#################################################################################################################################################
# Models
#################################################################################################################################################


###############################
# Logistic Regression
logreg = linear_model.LogisticRegression(max_iter=2000, class_weight="balanced", solver="sag")

# Grid search cross validation
parameters = {
    "C": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000],
    "multi_class": ["multinomial", "ovr"],
}

grid_search = GridSearchCV(logreg, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

logreg = grid_search.best_estimator_
print(logreg.score(test_X.values, test_y.values))

# Plots
plotTrainTestLines("Logistic Regression(" + str(number_of_columns) + " features)", logreg, train_X_np, train_y_np,
                   test_X.values, test_y.values)
clfs = {"LR": logreg}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs)

###############################
# Logistic regression

logreg2 = linear_model.LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)

# Grid search cross validation
parameters = {
    "C": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000],
}

grid_search = GridSearchCV(logreg2, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

logreg2 = grid_search.best_estimator_
print(logreg2.score(test_X.values, test_y.values))

# Plots
plotTrainTestLines("Logistic Regression(" + str(number_of_columns) + " features)", logreg2, train_X_np, train_y_np,
                   test_X.values, test_y.values)

clfs = {"LR": logreg2}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs)

###############################
# KNN

knn = KNeighborsClassifier(n_neighbors=5)

# Grid search cross validation
parameters = {
    "n_neighbors": [3, 5, 10, 20, 50, 100, 200],
    "weights": ["uniform", "distance"]
}

grid_search = GridSearchCV(knn, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

knn = grid_search.best_estimator_
print(knn.score(test_X.values, test_y.values))

# Plots
plotTrainTestLines("KNN(" + str(number_of_columns) + " features)", knn, train_X_np, train_y_np, test_X.values,
                   test_y.values)

# Get probabilities from model
probs = knn.predict_proba(test_X.values)
# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, test_y.values)
clfs = {}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs, {"KNN": prec_rec_dict})

###############################
# Bernoulli Naive Bayes

bernulli_NB = BernoulliNB(binarize=None)

f1s = cross_val_score(bernulli_NB, train_X_np, train_y_np, cv=5, scoring='f1_micro')
print(f1s)

bernulli_NB.fit(train_X_np, train_y_np)
print(bernulli_NB.score(test_X.values, test_y.values))

# Plots
plotTrainTestLines("Bernoulli NB(" + str(number_of_columns) + " features)", bernulli_NB, train_X_np, train_y_np,
                   test_X.values, test_y.values)

# Get probabilities from model
probs = bernulli_NB.predict_proba(test_X.values)
# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, test_y.values)
clfs = {}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs, {"NB": prec_rec_dict})

###############################
# Multinomial Naive Bayes

multi_NB = MultinomialNB()

f1s = cross_val_score(multi_NB, train_X_np, train_y_np, cv=5, scoring='f1_micro')
print(f1s)

multi_NB.fit(train_X_np, train_y_np)
print(multi_NB.score(test_X.values, test_y.values))

# Plots
# Get probabilities from model
probs = multi_NB.predict_proba(test_X.values)
# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, test_y.values)

plotTrainTestLines("Multinomial NB(" + str(number_of_columns) + " features)", multi_NB, train_X_np, train_y_np,
                   test_X.values, test_y.values)

clfs = {"LR": logreg}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs, {"NB": prec_rec_dict})
# exit(0)

###############################
# Support Vector Machine

svm = LinearSVC(max_iter=2000, dual=False)

# Grid search cross validation
parameters = {
    "C": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]
}

grid_search = GridSearchCV(svm, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

svm = grid_search.best_estimator_
print(svm.score(test_X.values, test_y.values))

# Plots
plotTrainTestLines("SVM(" + str(number_of_columns) + " features)", svm, train_X_np, train_y_np, test_X.values,
                   test_y.values)

clfs = {"SVM": svm}
plotPrecRecCurve(train_X_np, train_y_np, test_X, test_y, clfs)
