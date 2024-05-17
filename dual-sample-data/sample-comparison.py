# File for comparing with coworker's work in R Studio
# Ryan Magnuson rmagnuson@westmont.edu

# Setup
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
from scipy.stats.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data + Data Check
penguins = pd.read_csv("penguins.csv")
print(penguins.head())
print(penguins.shape)

# Remove column "sex"
print()
penguins = penguins.drop("sex", axis=1)
penguins = penguins.drop("rowid", axis = 1)
print(penguins.head())

# Plot (?)
plt.figure(figsize=(10,6))
sns.displot(
    data=penguins.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill"
)
plt.savefig("penguin_stats.png")
plt.close()

# Split data into training and testing
print()
np.random.seed(1128) # (?)
train, test = train_test_split(penguins,test_size=.20)

##
# Takes data input and splits it into target/predictor vars
# @param data: dataset
# @return target/predictor vars
def prep_data(data):
    df = data.copy()
    predictor = df.drop(["species"], axis=1) # "X"
    target = df["species"] # "y"

    return (predictor,target)

# remove n/a
train = train.dropna()
test = test.dropna()

# split testing/training into predictor/target vars
pred_train, target_train = prep_data(train)
pred_test, target_test = prep_data(test)
print(pred_train.shape, pred_test.shape, target_train.shape, target_test.shape)
print(penguins["species"].unique(), penguins["island"].unique())

# split columns into num/cat
print()
num_vars = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year'] # 'rowid',
cat_vars = ['species', 'island']

# NUM VARS #
print("TRAINING DATA")
print(train[num_vars].head())

# Density Plot

##
# Creates density plots for our numerical variables
# @param data dataset to be used
# @param m_cols number of columns
# @param m_rows number of rows
# @return density plot for different species base on numerical vars
def density_plot(data, m_cols, m_rows):
    fig, ax = plt.subplots(m_rows, m_cols, figsize=(10,10))

    for i in range(len(num_vars)):
        var = data[num_vars[i]]
        row = i // m_cols
        col = i % m_cols

        sns.kdeplot(x=var, hue=train["species"], fill=True, ax=ax[row,col])
        plt.tight_layout() #fix spacing

density_plot(train, 2, 3)
plt.savefig("density_plot.png")
plt.close()

# Correlation Matrix (!)
corrMatrix = train[num_vars].corr()
sns.heatmap(corrMatrix, annot=True)
plt.savefig("correlation_matrix.png")
plt.close()

# CAT VARS # WIP!
print()
print(train[cat_vars].head())

sns.catplot(x="species", hue="island", data=train, height=5, aspect=.9)
#plt.show()
plt.close()
#print(train.groupby(cat_vars)[["sex"]].count())

# p value
print()

##
# Computes correlation coefficients and corresponding p-vals
# @param var1 var
# @param var2 list of vars
# @return correlation coefficient and p-val
def p_val(var1, var2):
    for element in var2:
        print("CC and p-val between", var1, "and", element, "is:\n", pearsonr(train[var1], train[element]))

penguin_stats = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
p_val("flipper_length_mm", penguin_stats)

# Logistic Regression
print()
le = preprocessing.LabelEncoder() #label encoder object
pred_train["island"] = le.fit_transform(pred_train["island"])

LogReg = LogisticRegression(max_iter=1500)

##
# Trains and evaluates a model via cross validation on the cols of the data with selected indices
# @param cols list of columns/vars
# @return cross validation score
def check_column_score(cols):
    print("training w/ cols: " + str(cols))
    return cross_val_score(LogReg, pred_train[cols], target_train.values, cv=5)

combos = [['island', 'bill_length_mm', 'bill_depth_mm'],
          ['island', 'bill_length_mm', 'body_mass_g'],
          ['island', 'bill_depth_mm', 'body_mass_g']
]
for combo in combos:
    score = check_column_score(combo)
    print("CV score is: " + str(np.round(score, 3)))

# check found scores
print()
pred_test['island'] = le.fit_transform(pred_test['island'])

##
# Test the performance of the model trained on the cols of the data w/ selected indices
# @param cols list of cols/vars
# @return cross validation score
# def test_column_score(cols): # TODO: Uncomment here to use! (Error w/ "pytest")
#     print("training w/ cols: " + str(cols))
#     LogReg = LogisticRegression(max_iter=1500)
#     LogReg.fit(pred_train[cols], target_train)
#
#     return LogReg.score(pred_test[cols], target_test)
#
# for cols in combos:
#     score = test_column_score(cols)
#     print("The test score is: " + str(np.round(score, 3)))

# NOTE: Use predictors island, length, and depth (for now)

# MODELING
final_vars = ['island', 'bill_length_mm', 'bill_depth_mm']
le = preprocessing.LabelEncoder()
pred_train['island'] = le.fit_transform(pred_train['island'])
pred_test['island'] = le.fit_transform(pred_test['island'])

target_test_dc = le.fit_transform(target_test)

##
# creates graphs with decision regions corresponding to model
# @param model machine learning model
# @param pred predictor variable
# @param target target variable
# @return a graph with decision regions based on model
def plot_regions(model, pred, target):
    model.fit(pred,target)

    length = pred['bill_length_mm']
    depth = pred["bill_depth_mm"]

    grid_x = np.linspace(length.min(), length.max(), 501)
    grid_y = np.linspace(depth.min(), depth.max(), 501)

    # below are some awful variable names, I am so sorry. (it's for the grid)
    xx, yy = np.meshgrid(grid_x, grid_y)
    np.shape(xx), np.shape(yy)

    XX = xx.ravel()
    YY = yy.ravel()

    p = model.predict(np.c_[XX,YY])
    p = p.reshape(xx.shape)
    fig, ax = plt.subplots(1)

    # plot decision regions
    ax.contour(xx, yy, p, cmap="jet", alpha=.2)
    ax.scatter(length, depth, c=target, cmap="jet")
    ax.set(xlabel="bill_length_mm", ylabel="bill_depth_mm")


# Support Vector Machine
best_score = -np.inf # begin at -infinity
N = 30 # largest max 30
scores = np.zeros(N)

for i in range(1, N+1):
    SVM = svm.SVC(kernel="linear", C=i, random_state=1128)
    scores[i-1] = cross_val_score(SVM, pred_train[final_vars], target_train, cv=10).mean()
    if scores[i-1] > best_score:
        best_C = i
        best_score = scores[i-1]

print(best_C, best_score)

##
# creates a graph of values w/ scores
# @param max largest max value
# @param scores cross validation scores
# @param best_val best val from cross validation
# @param element parameter we're getting the best val for
# @return a graph of vals w/ scores
def best_graph(max, scores, best_val, element):
    fig, ax = plt.subplots(1)
    ax.scatter(np.arange(1, max+1), scores)
    ax.set(title="Best " + element + ": " + str(best_val))

best_graph(N, scores, best_C, "C")
plt.savefig("best_C.png")
plt.close()

# SVM w/ best C val
print()
SVM = svm.SVC(kernel="linear", C=best_C, random_state=1128)
model = SVM.fit(pred_train[final_vars], target_train)

target_test_pred = model.predict(pred_test[final_vars])
print(target_test_pred)

cm = confusion_matrix(target_test, target_test_pred)
print(cm, model.score(pred_test[final_vars], target_test))

plot_regions(SVM, pred_test[['bill_length_mm', 'bill_depth_mm']], target_test_dc) # plot results (ignore warning (?) )
plt.savefig("decision_regions.png")
plt.close()

# Random Forest Process
print("Please be patient...\n")
best_score = 0
best_n = None #stub for just in case
N_vals = [10,50,100,250,500]

for n in N_vals:
    F = RandomForestClassifier(n_estimators=n, random_state=1128)
    cvs = cross_val_score(F, pred_train[final_vars], target_train, cv=10).mean()
    if cvs > best_score:
        best_n = n
        best_score = cvs

print(best_n)
print(best_score)

# more confusing var names incoming
print()
m = RandomForestClassifier(best_n, random_state=1128)
model2 = m.fit(pred_train[final_vars], target_train)

target_test_pred = model2.predict(pred_test[final_vars]) #confusion matrix
c = confusion_matrix(target_test, target_test_pred)

print(c, model2.score(pred_test[final_vars], target_test))

plot_regions(m, pred_test[['bill_length_mm', 'bill_depth_mm']], target_test_dc)
plt.savefig("decision_regions_forest.png")
plt.close()

# KNN model
print()
best_score = -np.inf
N = 40 # largest max n_neighbors
scores = np.zeros(N)

for i in range(1, N+1):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores[i-1] = cross_val_score(knn, pred_train[final_vars], target_train, cv=10).mean()
    if scores[i-1] > best_score:
        best_n_neighbors = i
        best_score = scores[i-1]

print(best_n_neighbors, best_score)

best_graph(N, scores, best_n_neighbors, "n_neighbors") #make a graph
plt.savefig("best_n_neighbors.png")
plt.close()

# construct the knn model
print()
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(pred_train[final_vars], target_train)

target_test_pred = knn.predict(pred_test[final_vars])

cm = confusion_matrix(target_test, target_test_pred)
print(cm, knn.score(pred_test[final_vars], target_test))

#plot the knn graph
plot_regions(knn, pred_test[['bill_length_mm', 'bill_depth_mm']], target_test_dc)
plt.savefig("decision_regions_knn.png")
plt.close()