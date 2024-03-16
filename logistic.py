import numpy as np

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x):

    ##### WRITE YOUR CODE HERE #####
    return np.hstack((np.ones((x.shape[0],1)),x))
    #### END CODE ####

  def sigmoid(self, x):
    ##### WRITE YOUR CODE HERE ####
    z = x@self.w
    return  1/(1 + np.exp(-(z)))
    #### END CODE ####

  def cross_entropy(self, x, y_true):
    ##### WRITE YOUR CODE HERE #####
    y_pred = self.sigmoid(x)
    loss = np.divide(-(np.sum(y_true * np.log(y_pred) + (1- y_true) * np.log((1-y_pred)))),y_true.shape[0])
    return loss
    #### END CODE ####

  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    ##### WRITE YOUR CODE HERE #####
    x = self.add_ones(x)
    proba = self.sigmoid(x)
    return proba
    #### END CODE ####

  def predict(self,x):

    ##### WRITE YOUR CODE HERE #####
    probas = self.predict_proba(x)
    output = [1 if p >= 0.5 else 0 for p in probas]#convert the probalities into 0 and 1 by using a treshold=0.5
    return output
    #### END CODE ####

  def fit(self,x,y):

    # Add ones to x
    x = self.add_ones (x)

    # reshape y if needed
    y = y.reshape(-1,1)

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w  =  np.zeros((x.shape[1],1))

    for epoch in range(self.n_epochs):
      # make predictions
      y_pred = self.sigmoid(x)

      #compute the gradient
      grads = -1*((x.T)@(y - y_pred))/x.shape[0]


      #update rule
      self.w = self.w - self.lr * grads

      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)

      #if epoch%1000 == 0:
      print(f'loss for epoch {epoch}  : {loss}')

  def accuracy(self,y_true, y_pred):
    ##### WRITE YOUR CODE HERE #####
    acc = np.mean(y_true==y_pred)*100
    return print(acc)
    #### END CODE ####