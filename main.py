from linear import LinearRegression
from logistic import LogisticRegression
import numpy as np

# Start Train data for linear regression

X = np.linspace(0, 1, 10)
y = X + np.random.normal(0, 0.1, (10,))

X = X.reshape(-1,1)
y = y.reshape(-1,1)

# End of train data for linear regression

#Start Data for logistic regression

from sklearn.datasets import make_classification
X_x, y_y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = train_test_split(X_x,y_y)

#end logistic regression



linear = LinearRegression()
logistic = LogisticRegression(0.01,20)

def main():
    print("Which algorithms would you like to implement ? ")
    print("1. If Linear Regression choose 1")
    print("2. If Logistic Regression choose 2")
    user = input("Enter your choice")
    if user == "1":
       linear.fit(X,y)
    else:
       logistic.fit(X_train,y_train)
       logistic.accuracy(y_train,logistic.predict(X_train))

if __name__ == "__main__":
    main()
