import numpy as np
from scipy.stats import itemfreq

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, Xtrain, ytrain):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = Xtrain
    self.ytr = ytrain

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    num_class = self.ytr.shape[0]
    
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    Yknn = np.zeros(num_test, dtype = self.ytr.dtype)
    Yknnalex = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
#      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)   # broadcast in difference...
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
      min_index = np.argmin(distances) # get the index with smallest distance   
      
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      
     # ############## TRY K-NN algo... 
      k = 10
      sortdist = np.argsort(distances)
      myknn = sortdist[:k]   # prime k indici delle distance immagini piu vicine
      queue = sortdist[sortdist.shape[0]-k:sortdist.shape[0]] # last k ... should be very unprobable
      freq =itemfreq(self.ytr[myknn])
      freq_unpr = itemfreq(self.ytr[queue])
      maxindex = np.argmax(freq[:,1])
      Yknn[i] = freq[maxindex,0]
      ##########################################################
      
      ################### knn + penalize by queue prob...
      myarr = np.zeros(num_class, dtype = self.ytr.dtype)      
#      even_squares = [x ** 2 for x in nums if x % 2 == 0]
      for x in (self.ytr[myknn]):
          myarr[x] =  myarr[x]+1
      
      for x in (self.ytr[queue]):
          myarr[x] =  myarr[x]-1     
         
      #print(myarr)
      Yknnalex[i] = np.argmax(myarr)
      ###################################################
      
     
      print(Ypred[i], Yknn[i], Yknnalex[i]) 
      
    return Ypred, Yknn, Yknnalex