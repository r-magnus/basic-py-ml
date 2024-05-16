# File for testing beginner's scikit.
# Ryan Magnuson rmagnuson@westmont.edu

# Setup
from sklearn import datasets
import matplotlib.pyplot as plt

# Iris
iris = datasets.load_iris()
data = iris.data
print(data.shape)

# Digits (img ex)
digits = datasets.load_digits()
print(digits.images.shape)

print(plt.imshow(digits.images[-1], cmap=plt.cm.gray_r))

# scikit-learn
data = digits.images.reshape((digits.images.shape[0], -1)) #feature vector 8x8 -> 64
print(data)
