#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:09:59 2021

@author: evabes
"""

import pickle
import os


#Select configuration group (so far, 2)
config_group=1
output_folder='../Data/Settings/'


#In this file we prepare the list of parameterizations that we want to train the networks with
param_list=[]


if config_group==1:

    #†his group replicates the parameters of the paper, and tests what happens with DQN vs DDQN
    #what happens with two images types, and what happens with the 2 networ models
    
    #This parameterization is similar to the paper (sel_model=0 and render=0, with DQN (frequpdate=0)
    param_env={'incW':0.055,'decRel':50,'rOutMin':-0.5,'rOutMax':5,'rInMin':-255,'rInMax':255,'penalty':-10,'setRelevanceMap':False,'render':0}
    param_agent={'freqUpdate':0,'sel_model':0,'batch_size':250,'num_of_episodes':500,'steps_per_episodes':250,'mem_size':10000,'gamma':0.95,'alpha':0.001,'epsilon_max':0.99,'epsilon_step':7e-4}
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    
    #New: group, with network proposed by you
    param_agent['sel_model']=1
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=0 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    
    
     #New: group, with advantage network
    param_agent['sel_model']=2
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=0 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy()}
    param_list.append(param)
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    with open(output_folder+'param_listSingle1.pickle', 'wb') as handle:
        pickle.dump(param_list, handle)
    ###########################################################################

else:

    
   #This group replicates the parameters that you have putted in your code, and tests what happens with DQN vs DDQN
   #what happens with two images types, and what happens with the 2 networ models
  


  #This parameterization is similar to the paper (sel_model=0 and render=0, with DQN (frequpdate=0)
    param_env={'incW':0.055,'decRel':50,'rOutMin':0,'rOutMax':10,'rInMin':-255,'rInMax':255,'penalty':-10,'setRelevanceMap':False,'render':0}
    param_agent={'freqUpdate':0,'sel_model':0,'batch_size':64,'num_of_episodes':1000,'steps_per_episodes':40,'mem_size':10000,'gamma':0.95,'alpha':0.001,'epsilon_max':0.95,'epsilon_step':1e-3}
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    
    #New: group, with network proposed by you
    param_agent['sel_model']=1
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=0 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    
    #New: group, with advantage network
    param_agent['sel_model']=2
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=0 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    #New: change image model
    param_agent['freqUpdate']=0  #DQN
    param_env['render']=2 #New image model
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    
    param_list.append(param)
    
    #New
    param_agent['freqUpdate']=5 #DDQN
    param={'param_agent':param_agent.copy(),'param_env':param_env.copy(),'param_map':''}
    param_list.append(param)
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    
    with open(output_folder+'param_listSingle2.pickle', 'wb') as handle:
        pickle.dump(param_list, handle)
    ###########################################################################



