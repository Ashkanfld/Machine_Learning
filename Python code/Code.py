''' This code has been developed by Ashkan Fouladi (fooladiashkang@gmail.com)'''
''' 1) This program set a multi-linear regression for a dataset

2) Gradient descent is used as a method to minimize the objective function

3)To run this program, the user must input a dataset and 
a set of data for prediction in case of availibility, both in CSV format  

4) In the setting section user could define
the precision of stop criteria, the maximum cycle number and the learning rate (alpha)

5) If there is no need to predict: This program provides two CSV files  as an output;
first coefficients for each feature in the multi-linear regression model
and second The total cost and 
relative gap for each iteration 

6) If there is a given dataset to predict, in addition to the two mentioned files 
a CSV file is provided with the predicted results
 '''
 
# Modules
from numpy import random
import numpy as np
import statistics as st
import csv

# Variables
previousCost=0

#Setting
relativeGapAccuracy=1e-6
nCycle = 500
learningRate=1e-1

#Functions

#This function imports data from the dataset file (CSV)
def importData(CSV_file_path):
    with open(CSV_file_path) as f:
        content=csv.reader(f)
        next(content,None)
        next(content,None)
        x=[]
        y=[]
        for row in content:
            x.append(row[:-1])
            y.append(row[-1])
        m=len(y)
        n=len(x[0])
        for w in range(0,m):
            y[w]=float(y[w])
            z=np.zeros((m,n+1))
        for j in range(0,m):
            z[j][0]=1
            for i in range(1,n+1):
                z[j][i]=float(x[j][i-1])
    return z,y,n,m

#This Functions scale data using mean and standard deviation
def normalize(n,m,x,y):
    x1=np.array([])
    x2=np.array([])
    for i in range(1,n+1):
        s=[]
        for j in range(0,m):
            s.append(x[j][i])
        x1=np.append(x1,st.mean(s))
        x2=np.append(x2,st.stdev(s))
    y1=st.mean(y)
    y2=st.stdev(y)
    for w in range(0,m):
        y[w]=(y[w]-y1)/y2
        for z in range(1,n+1):
            x[w][z]=(x[w][z]-x1[z-1])/x2[z-1]
    return x1,x2,y1,y2

#This function set initial values for linear regression coefficients (Theta)
def initialGuess(e,n):
    a=[]
    for z in range(0,n+1):
        a.append(2*e*random.rand()-e)
    a=np.array(a)
    return a

#This function calculates the cost function 
def costFunction(x,y,b,m,pre):
    f=0
    for i in range(0,m):
        f+=(1/(2*m))*(b.dot(x[i])-y[i])**2
    rg=(pre-f)/f
    return f,rg

#This Function generates new coefficients using the gradient descent method
def gradient(x,y,b,m,n,alpha):
    new=np.array([])
    for i in range(0,n+1):
        s=0
        for j in range(0,m):
            s+=((-1*alpha)/m)*(b.dot(x[j])-y[j])*x[j][i]
        new=np.append(new,b[i]+s)
    for z in range(0,n+1):
        b[z]=new[z]

#This function predicts results based on regression
def findAnswer(file_path1,file_path2,x1,x2,y1,y2,b,n):
    with open(file_path1,'r',newline='') as f:
        content=csv.reader(f)
        next(content,None)
        next(content,None)
        x=[]
        for row in content:
            x.append(row)
        m=len(x)
        n=len(x[0])
        for w in range(0,m):
            z=np.zeros((m,n+1))
        for j in range(0,m):
            z[j][0]=1
            for i in range(1,n+1):
                z[j][i]=((float(x[j][i-1])-x1[i-1])/x2[i-1])
    ans=np.zeros(m)
    ans_normalized=np.zeros(m)
    for l in range(0,m):
        ans[l]=z[l].dot(b)
        ans_normalized[l]=y1+ans[l]*y2
    with open(file_path2,'w',newline='') as f:
        content=csv.writer(f)
        for d in range(0,len(ans_normalized)):
            content.writerow([ans_normalized[d]])
            
#This function report the total cost and relative gap at each iteration as a CSV file
def reportTotalCost(file_path,rg,tc,i):
    with open(file_path,'a',newline='') as f:
        myheader=['iteration','total cost','RG Gap']
        content=csv.DictWriter(f,fieldnames=myheader)
        if i==0:
            content.writeheader()
        content.writerow({'iteration':i,'total cost':tc,'RG Gap':abs(rg)})

#this Function reports final coefficients as a CSV file
def reportTheta(file_path,b):
    with open(file_path,'w',newline='') as f:
        content=csv.writer(f)
        content.writerow(b)
            
def main():
    for i in range(0,nCycle):
        if i==0:
            xData,yData,noFeatures,noTrials=importData('')
            x_mean,x_std,y_mean,y_std=normalize(noFeatures,noTrials,xData,yData)
            theta=initialGuess(1,noFeatures)
            (jVal,RG)=costFunction(xData,yData,theta,noTrials,previousCost)
            reportTotalCost('', RG, jVal, i)
        else:
            gradient(xData,yData,theta,noTrials,noFeatures,learningRate)
            (jVal,RG)=costFunction(xData,yData,theta,noTrials,jVal)
            print('Cost function at iteration number %i is %f'%(i,jVal))
            reportTotalCost('', RG, jVal, i)
            if RG>=0 and RG<relativeGapAccuracy:
                break
    reportTheta('',theta)
    print('Final Theta is ',theta)
    findAnswer('','',x_mean,x_std,y_mean,y_std,theta,noFeatures)

main()            