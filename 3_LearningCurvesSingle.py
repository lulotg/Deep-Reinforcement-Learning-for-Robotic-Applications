#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:09:06 2021

@author: evabes
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def identify_files(folder):
    
    files=glob.glob(folder_train+"*.names")
    files.sort(key=os.path.getmtime)
    hm=len(files)
    print(files)   
    
    identifiers=[]
    for i in range(hm):
        f=files[i]
        a=f.split('/')
        #print(a)
        hma=len(a)
        b=a[hma-1]
        #print(b)
        c=b.split('.')
        print(c)
        d=c[0]
        identifiers.append(d)
    #print(d)
    return identifiers


def load_draw(folder,strsel,c,vs_time):
    
    if vs_time:
        x = np.genfromtxt(folder+'timeBuffer_'+strsel+'.csv', delimiter=',')
    else:
        x = np.genfromtxt(folder+'episodeBuffer_'+strsel+'.csv', delimiter=',')
    reward_buffer = np.genfromtxt(folder+'rewardBuffer_'+strsel+'.csv', delimiter=',')
    filtered_reward_buffer = np.genfromtxt(folder+'filteredRewardBuffer_'+strsel+'.csv', delimiter=',')
 
    #print(c)
    plt.plot(x,reward_buffer,color=c,alpha=0.2)
    h=plt.plot(x,filtered_reward_buffer,color=c,label=strsel)
    plt.grid(True, which = 'both')
    if vs_time==False:
        plt.title("Reward per episode")
        plt.xlabel("Episodes", fontsize=14)
        plt.xlim([1,len(x)])
    else:
        plt.title("Reward per computation time")
        plt.xlabel("Computation Time (s)", fontsize=14)
        
    plt.ylabel("Reward", fontsize=14)
    
    return h




folder_train='../Data/Single1/TrainingCurves/'

identifiers=identify_files(folder_train)
hm=len(identifiers)

random_colors=False

if random_colors:
    colors=np.random.rand(hm,3)
else:
    #I have only put 8 colors
    colors=[[1,0,0],[0,1,0],[0,0,1],[1,0,1],[1,1,0],[0,1,1],[0.5,0.5,0.5],[0,0,0],[0.8,0.2,0.9],[1-0.8,1-0.2,1-0.9],[0.5,0.0,0.1],[0.2,0.9,0.3]]

plt.style.use('default')
plt.figure(figsize=(12, 6))   
hs=[]
for i in range(hm):
    h=load_draw(folder_train,identifiers[i],tuple(colors[i]),False)
    hs.append(h)

plt.legend()
plt.show()

plt.style.use('default')
plt.figure(figsize=(12, 6))  
hs=[] 
for i in range(hm):
    h=load_draw(folder_train,identifiers[i],tuple(colors[i]),True)
    hs.append(h)

plt.legend()
plt.show()





