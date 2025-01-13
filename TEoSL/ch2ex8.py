import numpy as np
import pathlib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# Ex 2.8
# Compare the classification performance of linear regression and k–nearest neighbor classification
# on the zipcode data. In particular, consider only the 2’s and 3’s, and k = 1, 3, 5, 7 and 15. 
# Show both the training and test error for each choice. The zipcode data are available from the 
# book website www-stat.stanford.edu/ElemStatLearn.

# results:
# Test error rate of Linear Regression is 4.12%
# Train error rate of Linear Regression is 0.58%
# k-NN Model: k is 1, train/test error rates are 0.00% and 2.47%
# k-NN Model: k is 2, train/test error rates are 0.58% and 2.47%
# k-NN Model: k is 3, train/test error rates are 0.50% and 3.02%
# k-NN Model: k is 4, train/test error rates are 0.43% and 2.75%
# k-NN Model: k is 5, train/test error rates are 0.58% and 3.02%
# k-NN Model: k is 6, train/test error rates are 0.50% and 3.02%
# k-NN Model: k is 7, train/test error rates are 0.65% and 3.30%
# k-NN Model: k is 8, train/test error rates are 0.58% and 3.30%
# k-NN Model: k is 9, train/test error rates are 0.94% and 3.57%
# k-NN Model: k is 10, train/test error rates are 0.79% and 3.57%
# k-NN Model: k is 11, train/test error rates are 0.86% and 3.57%
# k-NN Model: k is 12, train/test error rates are 0.72% and 3.57%
# k-NN Model: k is 13, train/test error rates are 0.86% and 3.85%
# k-NN Model: k is 14, train/test error rates are 0.86% and 3.85%
# k-NN Model: k is 15, train/test error rates are 0.94% and 3.85%

# get relative data folder
PATH = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PATH.joinpath("data").resolve()

# get original data
train = np.genfromtxt(DATA_PATH.joinpath("zip_train_2and3.csv"), dtype=float, delimiter=',', skip_header=True)
test = np.genfromtxt(DATA_PATH.joinpath("zip_test_2and3.csv"), dtype=float, delimiter=',', skip_header=True)

# prepare training and testing data
x_train, y_train = train[:, 1:], train[:, 0]
x_test, y_test = test[:, 1:], test[:, 0]

# python slice notation:
#   (not to be confused with python's slice object: "slice()" e.g. a[slice(start, stop, step)] is equivalent to a[start:stop:step])

# a[start:stop]  # items start through stop-1 *(:stop value represents the first value that is NOT in the selected slice.)
# a[start:]      # items start through the rest of the array
# a[:stop]       # items from the beginning through stop-1
# a[:]           # a copy of the whole array

# a[start:stop:step] # start through stop, by step

# start or stop may be a negative number, which means it 
# counts from the end of the array instead of the beginning. So:
# a[-1]    # last item in the array
# a[-2:]   # last two items in the array
# a[:-2]   # everything except the last two items

# Similarly, step may be a negative number:

# a[::-1]    # all items in the array, reversed
# a[1::-1]   # the first two items, reversed
# a[:-3:-1]  # the last two items, reversed
# a[-3::-1]  # everything except the last two items, reversed


print("train:")
print(train)

# print("train[:, 0]")
# print(train[:, 0])

# print("train[:, 1:]")
# print(train[:, 1:])

# print("train[:2, 0]")
# print(train[:2, :])

print("x_train:")
print(x_train)
print("y_train:")
print(y_train)

# for classification purpose
# we assign 1 to digit '3' and 0 to '2'
y_train[y_train == 3] = 1
y_train[y_train == 2] = 0
y_test[y_test == 3] = 1
y_test[y_test == 2] = 0


print("y_train:")
print(y_train[:10])

# a utility function to assign prediction
def assign(arr):
    arr[arr >= 0.5] = 1
    arr[arr < 0.5] = 0

# a utility function to calculate error rate
# of predictions
def getErrorRate(a, b):
    if a.size != b.size:
        raise ValueError('Expect input arrays have equal size, a has {}, b has {}'.
                        format(a.size, b.size))

    if a.size == 0:
        raise ValueError('Expect non-empty input arrays')

    return np.sum(a != b) / a.size

# Linear Regression
reg = LinearRegression().fit(x_train, y_train)
pred_test = reg.predict(x_test)
assign(pred_test)
print('Test error rate of Linear Regression is {:.2%}'
    .format(getErrorRate(pred_test, y_test)))

pred_train = reg.predict(x_train)
assign(pred_train)
print('Train error rate of Linear Regression is {:.2%}'
    .format(getErrorRate(pred_train, y_train)))

# run separate K-NN classifiers
for k in range(1, 16):
    # fit the model
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)

    # test error
    pred_knn_test = neigh.predict(x_test)
    assign(pred_knn_test)
    test_error_rate = getErrorRate(pred_knn_test, y_test)
    # train error
    pred_knn_train = neigh.predict(x_train)
    assign(pred_knn_train)
    train_error_rate = getErrorRate(pred_knn_train, y_train)

    print('k-NN Model: k is {}, train/test error rates are {:.2%} and {:.2%}'
        .format(k, train_error_rate, test_error_rate))