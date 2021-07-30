# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class MOG():
    def __init__(self,numOfGauss=3,meanVal=0.0, varVal = 1.0, BG_thresh=0.6, lr=0.01, height=240, width=320):
        self.numOfGauss=numOfGauss
        self.BG_thresh=BG_thresh
        self.lr=lr
        self.height=height
        self.width=width
        self.mus=np.zeros((self.height,self.width, self.numOfGauss,3)) ## assuming using color frames
        #self.mus=np.zeros((self.height,self.width, self.numOfGauss)) ## assuming using gray-scale frames
        self.sigmaSQs=np.zeros((self.height,self.width,self.numOfGauss)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas=np.zeros((self.height,self.width,self.numOfGauss))
        self.currentBG=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array(meanVal*self.numOfGauss) ##assuming a 400ppm for CO2 mean 
                #self.mus[i,j]=np.array(meanVal) ##assuming a 400ppm for CO2 mean 
                self.sigmaSQs[i,j]=[varVal]*self.numOfGauss
                self.omegas[i,j]=[1.0/self.numOfGauss]*self.numOfGauss
                
    def reorder(self):
        BG_pivot=np.zeros((self.height,self.width),dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                BG_pivot[i,j]=-1
                ratios=[]
                for k in range(self.numOfGauss):
                    ratios.append(self.omegas[i,j,k]/np.sqrt(self.sigmaSQs[i,j,k]))
                indices=np.array(np.argsort(ratios)[::-1])
                self.mus[i,j]=self.mus[i,j][indices]
                self.sigmaSQs[i,j]=self.sigmaSQs[i,j][indices]
                self.omegas[i,j]=self.omegas[i,j][indices]
                cummProb=0
                for l in range(self.numOfGauss):
                    cummProb+=self.omegas[i,j,l]
                    if cummProb>=self.BG_thresh and l<self.numOfGauss-1:
                        BG_pivot[i,j]=l
                        break
                ##if no background pivot is made the last one is foreground
                if BG_pivot[i,j]==-1:
                    BG_pivot[i,j]=self.numOfGauss-2
        return BG_pivot
    
    def updateParam(self, curFrame, BG_pivot):
        labels=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                X=curFrame[i,j]
                match=-1
                for k in range(self.numOfGauss):
                    CoVarInv=np.linalg.inv(self.sigmaSQs[i,j,k]*np.eye(3))
                    X_mu=X-self.mus[i,j,k]
                    dist=np.dot(X_mu.T, np.dot(CoVarInv, X_mu))
                    if dist<6.25*self.sigmaSQs[i,j,k]: #6.25 = 2.5^2; 9.0=3.0^2
                        match=k
                        break
                if match!=-1:  ## a match found
                    ##update parameters
                    self.omegas[i,j]=(1.0-self.lr)*self.omegas[i,j]
                    self.omegas[i,j,match]+=self.lr
                    rho=self.lr * multivariate_normal.pdf(X,self.mus[i,j,match],np.linalg.inv(CoVarInv))
                    self.sigmaSQs[i,j,match]=(1.0-rho)*self.sigmaSQs[i,j,match]+rho*np.dot((X-self.mus[i,j,match]).T, (X-self.mus[i,j,match]))
                    self.mus[i,j,match]=(1.0-rho)*self.mus[i,j,match]+rho*X
                    ##label the pixel
                    if match>BG_pivot[i,j]:
                        labels[i,j]=250
                else:
                    self.mus[i,j,-1]=X
                    labels[i,j]=250
        return labels
    
    def extractCurrentBG(self):
        for i in range(self.height):
            for j in range(self.width):
                self.currentBG[i, j] = np.dot(self.omegas[i, j], self.mus[i, j, :, 0])
                
            
    def streamer(self):
        ## initialize pixel gaussians
        data_array = np.genfromtxt('630091_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        #data_array = np.genfromtxt('630094_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        TotalNumFrames = np.size(data_array, 0)
        data_array = np.expand_dims(data_array, axis=1)
        for fr in range(TotalNumFrames):
            frame=data_array[fr, :, :]
            
            print("number of frames: ", fr)
            print(frame[0, :])
            BG_pivots=self.reorder()
            labels=self.updateParam(frame,BG_pivots)
            #print(labels)
            #print(np.sum(labels[0, :] > 0.0))
            plt.figure(1)
            plt.plot(frame[0, :])
            plt.plot(labels[0,:], 'r') #red color
            plt.axis([0, 17500, 0, 1600])
            
            plt.show()
            
            # # extract plot current BG
            # self.extractCurrentBG()
            # plt.figure(2)
            # plt.plot(self.currentBG[0, :]) 
            # plt.axis([0, 17500, 400, 1600])
            # plt.show()
        
        
        
def main():
    subtractor=MOG(numOfGauss=4,meanVal=400.0, varVal = 200.0, BG_thresh=0.6, lr=0.1, height=1, width=17280) # note to change width accordingly
    subtractor.streamer()
    
    
if __name__=='__main__':
    main()
