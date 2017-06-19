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

#function to compute X.XT
def generateA(Xi):
	#print "Xi:",Xi	
	Xmatrix=np.matrix(Xi, dtype=float)
	XiT = np.insert(Xmatrix, 0, 1, axis=1)
	#print "XiT:",XiT
	Xi = XiT.T
	mul_a = np.dot(Xi,XiT)
	return mul_a

#function to compute XY
def generateB(Yi,Xi):
	Xmatrix=np.matrix(Xi, dtype=float)
	XiT = np.insert(Xmatrix, 0, 1, axis=1)
	Ymatrix=np.matrix(Yi, dtype=float)
	Xi = XiT.T
	mul_b = np.dot(Xi,Ymatrix)
	return mul_b


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength
  

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)
  
  #compute X.XT for each line 	
  A=yxlines.map(lambda line: generateA(line[1:]))

  #compute XY for each line
  B=yxlines.map(lambda line: generateB(line[0],line[1:]))
  
  #print A.first()
  #print B.first()

  A_sum=A.reduce(lambda Xi, Xj: np.add(Xi,Xj)) #summation of X.Xt
  B_sum=B.reduce(lambda Xi, Yi: np.add(Xi,Yi)) #summation of XY
  #print A_sum
  #print B_sum
  
  #Finally, compute beta using the formula A(inverse)*B
  beta=np.dot(np.linalg.inv(A_sum),B_sum)

  #save the output onto a text file
  np.savetxt('yxlinoutput.txt',beta)


  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff

  sc.stop()
