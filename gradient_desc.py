# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
import fileinput
from numpy.linalg import inv

from pyspark import SparkContext


def getX(Xi):
	Xi[0]=1.0
	return Xi

def getY(Yi):
	Ymatrix=np.asmatrix(Yi, dtype=float)
	return Ymatrix


if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile> <alpha> <iterations>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  #Fetch alpha and iteration values from command line
  alpha = float(sys.argv[2])
  iterations = int(sys.argv[3])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength
  
  #collect all y values into a matrix
  Y=np.asmatrix(yxlines.map(lambda line: line[0]).collect()).astype(float).T
  
  #collect all x values into a matrix
  X=np.asmatrix(yxlines.map(lambda line: getX(line)).collect()).astype(float)
  #print Y
  #print X

  #Calculate the order of beta based on order of X
  beta_order = X.shape[1]  
  
  #Initialize beta values to 1
  init_beta= [1 for i in range(0,beta_order)]
  beta_iterations = []

  flag = False #to check for convergence
  
  beta_prev=init_beta  
  
  for iteration in range(0,iterations):     
      #Gradient descent equation (b + Xt.alpha.(Y-Xb))            
      beta = np.matrix(beta_prev).T
      xbeta = np.dot(X,np.matrix(beta)) 
      diff = np.subtract(Y,xbeta)
      betaFinal = (np.add(beta,np.multiply(np.dot(X.T,diff),alpha)))
      beta_curr = [] 
        
      #Add beta values to a list
      for val in betaFinal.tolist():
          beta_curr.append(val[0])
        
      #Check if previous beta values are same as current.
      if beta_prev == beta_curr:
          flag = True
          print('Converged at '+str(iteration)+'th iteration')
            
          print ("beta: ")
          for coeff in beta_curr:       
              print (coeff)
          break
        
      #Update beta values
      beta_prev = beta_curr
      beta_iterations = beta_curr
    
  #Check for convergence
  if not flag:
      print('Not convereged for '+ str(iterations)+'iterations') 

  
  #print "beta: "
  #for coeff in beta:
      #print coeff

  sc.stop()
