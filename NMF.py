#!/bin

#Author: Jayant Gupta
#Date: Jan, 13, 2014

# Theoretical Source:
	# Algorithms for Non-negative Matrix Factorization. - Daniel D. Lee, H Sebastian  Seung

# Cost function : |V - WH|^2 minimizing this function subject to the constraints W,H >= 0.
# First use simple gradient descent function to arrive at a local optima.
# if ( a = N*M )
# W = N*r
# H = r*M

#Observations : As the number of dimensions increases the approximation improves (pretty intuitive).


import numpy as np
import random 

def NMF(matrix):

	#!Caution, I haven't done any basic dimensionality checks.

	N=len(matrix)
	M=len(matrix[0])
	r = input("Give the input dimension: ")
	eta = input("Give the descent parameter: ")

	# Initialization : each and every value with random numbers.
	a=[1.0*i for i in range(10)]
	W = np.array([random.sample(a,r) for j in range(N)])
	H = np.array([random.sample(a,M) for j in range(r)])
	WH = np.dot(W,H)
	c=cost(matrix, WH)

	for i in range(50000):
#i=0;
#	while(c > 0.1):
		print("loop: "+str(i))
		print(W)
		print(H)
		
		# Iteratively calculating the value of H and W parameters.

		H = H + eta*(np.dot(np.transpose(W), matrix) + -1*np.dot(np.transpose(W),WH))
		W = W + eta*(np.dot(matrix, np.transpose(H)) + -1*np.dot(WH,np.transpose(H)))

#		H = np.divide(np.dot(np.transpose(W),matrix), np.dot(np.transpose(W),WH))
#		print(H)
#		H  = np.dot(H, np.divide(np.dot(np.transpose(W),matrix), np.dot(np.transpose(W),WH)))
#		W  = np.dot(W, np.divide(np.dot(matrix,np.transpose(H)), np.dot(WH,np.transpose(H))))
		
		WH = np.dot(W,H)
		c = cost(matrix, WH)
		print("cost="+str(c))
#		i+=1

	print("\n\n Final results")
	print(W)
	print(H)
	print("\nEstimated matrix")
	print(np.dot(W,H))
	print("\n Given Matrix")
	print(np.array(matrix))
	print("\nGiven eta:"+str(eta))
	print(c)

def cost(V,WH):
	N=len(V)
	M=len(V[0])
	c=0;
	for i in range(N):
		for j in range(M):
			c+=(V[i][j]-WH[i][j])**2

	return c

if __name__ == "__main__":
	print("\n~~~ Hasta La Vista ~~~ \n Welcome to Non-negative matrix factorization")	
	a=[[],[],[],[]]
	a[0]=[2,0,8,0]
	a[1]=[7,3,0,4]
	a[2]=[0,3,1,9]
	a[3]=[0,3,0,0]
	NMF(a)
