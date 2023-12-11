import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


VERBOSE = False

#read in dataset
df = pd.read_csv("kaggle/input/external-data/train_preprocessed_fixed.csv")

X,y = df["text"],df["label"]
data_train, data_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5,
                             max_features=1000,ngram_range=(1,3))

X_train = vectorizer.fit_transform(data_train)
X_test = vectorizer.transform(data_test)

feature_names = vectorizer.get_feature_names_out()

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(["0","1"])
ax.yaxis.set_ticklabels(["0","1"])
_ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
)

plt.show()

print("HUMAN PREDICTIVE WORDS/PHRASES")
print("---------------------------")
average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
top_human_indices = np.argsort(average_feature_effects[0])[-20:][::-1]
human_predictive_ngrams = feature_names[top_human_indices]
for i in top_human_indices:
    print(feature_names[i],average_feature_effects[0][i])

print("AI PREDICTIVE WORDS/PHRASES")
print("---------------------------")
top_ai_indices = np.argsort(average_feature_effects[0])[:20]
ai_predictive_ngrams = feature_names[top_ai_indices]
for i in top_ai_indices:
    print(feature_names[i],average_feature_effects[0][i])

print("Words/phrases most likely found in an human-generated essay:")
print(human_predictive_ngrams)
print("Words/phrases most likely found in a AI-generated essay:")
print(ai_predictive_ngrams)

if VERBOSE:
    print(df.columns)
    print(df.shape)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)