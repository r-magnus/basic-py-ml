# Specfiic scikit tutorial for text data
# Ryan Magnuson trmagnuson@westmont.edu

# NOTE: this file seems like it may be very good for review.

"""
NOTES:
  - R has an easy Confidence Interval (CI)
  - R can return a correlation table
    * this is where the relationships between features are measured
    * we should achieve something similar with Py
  -
"""

# Setup
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Load
print("LOAD:")
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
print(twenty_train.target[:10])

# Tokenizing
print("\nTOKENIZING:")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

print(X_train_counts.shape)

# Transformer
print("\nTRANSFORMER:")
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)

# Classifier
print("\nCLASSIFIER:")
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

# Pipeline
print("\nPIPELINE:")
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

print(text_clf.fit(twenty_train.data, twenty_train.target))

# Performance
print("\nPERFORMANCE:")
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)

print(np.mean(predicted == twenty_test.target))

# Improving Accuracy (Performance)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

# Metric Report
print("\nMETRICS:")
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))

# Param Tuning (slow)
# print()
# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (1e-2, 1e-3),
# }
#
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
#
# print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])
#
# for param_name in sorted(parameters.keys()):
#   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

