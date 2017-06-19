# Linear-Regression-Python-Spark

Contents of the Zip file:


1) linreg.py: Python-spark implementation of the Ordinary Least Square estimate of beta coefficients.

2) gradient_desc.py: Python-spark implementation of Gradient Descent version of Linear Regression.

3) yxlin.out: Output file obtained on running linreg.py for the input file yxlin.csv.

4) yxlin2.out: Output file obtained on running linreg.py for the input file yxlin2.csv.

5) yxlin_grad.out: Output file obtained on running gradient_desc.py for the input file yxlin.csv.

6) yxlin2_grad.out: Output file obtained on running gradient_desc.py for the input file yxlin2.csv

7) README.txt: This document.


Python version: 2.7.6

Spark version: 1.6.0



Algorithm:


1) linreg.py: Estimation of beta coefficients using Ordinary Least Square equation given as: Beta=((Xt.X)^-1)*(Xt.Y). Two mappers were used. One to obtain the matrix X.Xt for each line and another to obtain the matrix X.Y for each line in the csv. This was achieved through the functions generateA() and generateB(). Two reducers were used to obtain the summation of the values produced by the each of the mappers, and the outputs were read into the matrices A_sum and B_sum respectively. Finally, the value for beta was obtained by taking an inverse of the A_sum matrix and multipying it with the B_sum atrix. The numpy library of python was extensively used to perform mathematical operations such as matrix multiplication,addition, transpose and inverse.



2) gradient_desc.py: An alternate, iterative method for implementing linear regresion using the euqation: Beta=Beta+Xt.alpha.(Y-XBeta). Two mapper funtions were used to collect the X and Y values separately and read them as matrices. Beta was intitialized to a matrix containing all 1s. For a fixed number of iterations, the value of beta was computed, and updated at each iteration from the starting values of all 1s. The step size was denoted by alpha, and the values for alpha and number of iterations were read as command-line arguments. (For experiment, the value for alpha was assumed to be 0.001). At the end of each iteration, the values obtained for beta were compared with the previous values. If they became equal to each other, then the point of convergence has been reached, and the iteration is recorded. This corresponds to the beta coefficients obtained in the negative direction of the gradient.



Steps to Run the Program:


Copy the .py file and the input files to hdfs, and use an scp command to fetch them to the user home directory of the dsba cluster. Run the spark submit command as follows:


1) For OLS linear regression: 
$spark-submit linreg.py /user/<username>/input/yxlin.csv 



2) For Gradient Descent: 
$spark-submit gradient_desc.py /user/<username>/input/yxlin.csv 0.001 3000

