import re
import string

import numpy as np


    
def sigmoid(z): 
   
    h = 1/(1 + np.exp(-z))
    
    return h
    
def gradientDescent(x, y, theta, alpha, num_iter):
    
    
    m = len(x)
  
    for i in range(0, num_iter):
      
        z = np.dot(x,theta)
        h = sigmoid(z)
               
        J = (-1/m)*(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h)))
        theta = theta - (alpha/m)*np.dot(x.T, h-y)
        
    J = float(J)
    
    return J, theta

