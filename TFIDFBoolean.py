from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from src.helper import *

tweets_path = "D:/tweets.csv"
positive_words_path = 'positive-words.txt'
negative_words_path = 'negative-words.txt'
emoticons_path = "emoticons.txt"

# Read data and preprocess
tweet_data = read_data(tweets_path)
tweet_data["Tweet"] = tweet_data["Tweet"].apply(preprocess_tweet)

# Read positive/negative words
positive_words = read_words(positive_words_path)
negative_words = read_words(negative_words_path)

tweet_data = create_base_features(tweet_data, positive_words, negative_words, True)

# Get all feature columns
base_features = list(tweet_data.columns.values)
print(base_features)
base_features.remove('Category')
base_features.remove('Tweet')
# print(base_features)

# Tf-idf vectorizer
vect = TfidfVectorizer(max_features=6000, use_idf=True)
vect.fit(tweet_data["Tweet"])

simple_train_dtm = vect.transform(tweet_data["Tweet"])

std = simple_train_dtm.toarray()
X = std

# Encode labels
le = LabelEncoder()
tweet_data["Category"] = le.fit_transform(tweet_data["Category"])
tweet_data["Category"].head()

y = np.asarray(tweet_data["Category"])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)
# print(len(train_X))
# print(train_X.head())


# Perform PCA
number_of_columns = 150
pca = PCA(n_components=number_of_columns)

pca.fit(train_X)

train_X = pca.transform(train_X)

test_X = pca.transform(test_X)

train_X_np, train_y_np = undersample(train_X, train_y)

#################################################################################################################################################
#################################################################################################################################################
# Models
#################################################################################################################################################

###############################
# Logistic Regression
logreg = linear_model.LogisticRegression(max_iter=1000, class_weight="balanced", solver="sag")

# Grid search cross validation
parameters = {
    "C": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000],
    "multi_class": ["multinomial", "ovr"]
}

grid_search = GridSearchCV(logreg, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

logreg = grid_search.best_estimator_
print(logreg.score(test_X, test_y))

# Plots
plotTrainTestLines("Logistic Regression(" + str(number_of_columns) + " features)", logreg, train_X_np, train_y_np,
                   test_X, test_y)

clfs = {"LR": logreg}
plotPrecRecCurve2(train_X_np, train_y_np, test_X, test_y, clfs)

###############################
# Logistic Regression
logreg2 = linear_model.LogisticRegression(solver="liblinear", max_iter=1000)

# Grid search cross validation
parameters = {
    "C": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]
}

grid_search = GridSearchCV(logreg2, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

logreg2 = grid_search.best_estimator_
print(logreg2.score(test_X, test_y))

# Plots
plotTrainTestLines("Logistic Regression2(" + str(number_of_columns) + " features)", logreg2, train_X_np, train_y_np,
                   test_X, test_y)

clfs = {"LR": logreg2}
plotPrecRecCurve2(train_X_np, train_y_np, test_X, test_y, clfs)

###############################
# KNN

knn = KNeighborsClassifier(n_neighbors=5)

# Grid search cross validation
parameters = {
    "n_neighbors": [3, 5, 10, 20, 50, 100],
    "weights": ["uniform", "distance"]
}

grid_search = GridSearchCV(knn, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

knn = grid_search.best_estimator_
print(knn.score(test_X, test_y))

# Plots
plotTrainTestLines("KNN(" + str(number_of_columns) + " features)", knn, train_X_np, train_y_np, test_X, test_y)

# Get probabilities from model
probs = knn.predict_proba(test_X)
# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, test_y)

clfs = {}
plotPrecRecCurve2(train_X_np, train_y_np, test_X, test_y, clfs, {'KNN': prec_rec_dict})

###############################
# Bernoulli Naive Bayes

bernulli_NB = BernoulliNB(binarize=None)

f1s = cross_val_score(bernulli_NB, train_X_np, train_y_np, cv=5, scoring='f1_micro')
print(f1s)

bernulli_NB.fit(train_X_np, train_y_np)
print(bernulli_NB.score(test_X, test_y))

# Plots
plotTrainTestLines("Bernoulli NB(" + str(number_of_columns) + " features)", bernulli_NB, train_X_np, train_y_np, test_X,
                   test_y)

###############################
# Support Vector Machine

svm = LinearSVC(max_iter=2000, dual=False)

# Grid search cross validation
parameters = {
    "C": [0.0001, 0.01, 0.05, 0.1, 0.5, 1, 10]
}

grid_search = GridSearchCV(svm, scoring="f1_micro", cv=5, param_grid=parameters)
grid_search.fit(train_X_np, train_y_np)

print(grid_search.best_score_)
print(grid_search.best_params_)

svm = grid_search.best_estimator_
print(svm.score(test_X, test_y))

# Plots
plotTrainTestLines("SVM(" + str(number_of_columns) + " features)", svm, train_X_np, train_y_np, test_X, test_y)

clfs = {"SVM": svm}
plotPrecRecCurve2(train_X_np, train_y_np, test_X, test_y, clfs)

###################
# Test Predictions

test_path = "test_tweets.csv"
test_data = read_data(test_path)
test_data["Tweet"] = test_data["Tweet"].apply(preprocess_tweet)

le = LabelEncoder()
print(test_data)
test_data["Category"] = le.fit_transform(test_data["Category"])

print(test_data["Category"])
simple_train_dtm = vect.transform(test_data["Tweet"])
std = simple_train_dtm.toarray()

test_X = pca.transform(std)
print(logreg.predict(test_X))
