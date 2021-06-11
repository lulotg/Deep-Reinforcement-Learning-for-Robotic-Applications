#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:39:32 2021

@author: evabes
"""
import pandas as pd
import numpy as np
#import pushover
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Bibliotecas de NNs (Keras y Tensorflow)
from tensorflow.keras.backend import clear_session
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import glob
import os


# Import defined interactions with the environment.
import EnvironmentSingle as Environment
# Import agent actions and model.
#import AgentSingle_QLearning


def identify_files(folder):
    
    files=glob.glob(folder+"*.names")
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



#Plotting barplot with the information in the Data Frame 
def plot_barblox(df,title,ylabel):

    #matplotlib inline
    # Make the figures big enough for the optically challenged.
    fig = plt.figure()
    #box plot the numerical attributes
    #convert data frame to array for plotting
    plot_array = df.values
    plt.boxplot(plot_array)
    # Nice labels using attribute names on the x-axis
    plt.xticks(range(1,len(df.columns)+1),df.columns,rotation='vertical')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# Wrapper para seleccionar y componer el estado como las dos matrices
def do_step(env,action):    
    obs, reward, done, info = env.step(action)       
    state = np.dstack((obs['visited_map'],obs['importance_map']))    
    return state, reward, done, info
    
def reset(env):
    
    obs = env.reset()       
    state = np.dstack((obs['visited_map'],obs['importance_map']))
    
    return state    




def run_episode(N,env,model,draw):
    
    
    #state = np.dstack((obs['visited_map'],obs['importance_map']))
    state = env.render()           
    #position = obs['position']
  
   
    reward = 0
    
    if draw['simulate_one_case']:
        ax1=draw['ax1']
        writer=draw['writer']
        fig=draw['fig']
        video_folder=draw['video_folder']
        if not os.path.isdir(video_folder):
            os.makedirs(video_folder)
        video_file=video_folder+draw['video_file']
               
        with writer.saving(fig, video_file, 100):
            ax1.imshow(state) 
            plt.show()
            plt.pause(0.1)
            writer.grab_frame()
            for steps in range(N):
                if np.random.rand()<0.9:
                    # Predicción y accion #
                    q_values = model.predict(state[np.newaxis])
                    action = np.argmax(q_values[0])
                else:
                    action = np.random.randint(0,8)
              
                obs,rew,done,info = env.step(action)
                reward += rew
               
                #state = np.dstack((obs['visited_map'],obs['importance_map']))
                state = env.render()
                #position = obs['position']
                print("\rRewward on last actions has been: {0:.3f}".format(rew))
                ax1.imshow(state) 
                plt.show()
                plt.pause(0.1)
                writer.grab_frame()
    else:
     
        for steps in range(N):
            if np.random.rand()<0.9:
                # Predicción y accion #
                q_values = model.predict(state[np.newaxis])
                action = np.argmax(q_values[0])
            else:
                action = np.random.randint(0,8)
          
            obs,rew,done,info = env.step(action)
            reward += rew
           
            #state = np.dstack((obs['visited_map'],obs['importance_map']))
            state = env.render()
            #position = obs['position']
          
            
    return reward
        

# Kill all sessions running (for clear start point).
keras.backend.clear_session()

folder_net='../Data/Single1/Networks/'
folder_train='../Data/Single1/TrainingCurves/'
video_folder = "../Data/Single1/Video/"
video_file="Video1.gif"


#Files to analyze
identifiers=identify_files(folder_train)
#identifiers=[identifiers[7]]
hm=len(identifiers)


reduce_for_testing=False


if reduce_for_testing:
    #Number of episodes for each case
    episodes=10
    #Length of the episode
    N = 50
else:
    #Number of episodes for each case
    episodes=30 
    #Length of the episode
    N = 100


#It is important to know if we are just running one case (to plot graphics) or multiple (to obtain statistics)
if episodes>1 or hm> 1:
    simulate_one_case=False
else:
    simulate_one_case=True  
    
   
covers=[]
timemeans=[]
  

for k in range(hm):


    file = open(folder_train+'param_'+identifiers[k]+'.pickle', 'rb')
    param=pickle.load(file)
    file.close()
    
    param_env=param['param_env']
    param_env['setRelevanceMap']=False  #Si se pone a false, se ve mejor a donde no ir (creo que no influye en el reward)
    #param_env['render']=2  #Solo se usa para dibujar ahora, asi que aqui el mejor render es el que proponen ellos
    
    
    # Get the environment (this will get all methods).
    env = Environment.environment(param['param_map'],param_env)
    
    clear_session() 
    model = keras.models.load_model(folder_net+'BEST_NN_'+identifiers[k]+'.h5')
    
    all_metrics=[]
    
    grid_map = env.map
  
    asv_param=param['param_agent']
    print('============================================')
    print('File:'+ identifiers[k] + ' Render:'+str(env.whichrender)+' Model:'+str(asv_param['sel_model'])+' Freq:' +str(asv_param['freqUpdate']))
    print('============================================')
    
    for i in range(episodes):
    
        # We must reset the environment.
        obs=env.reset()
        env.set_test_mode(True)
        
  
        
        
        if simulate_one_case:
            #Prepare figures
            fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8))
            
            ax2.set_xticks(np.arange(env.standby.shape[1]))
            ax2.set_yticks(np.arange(env.standby.shape[0]))
            ax2.grid(True, linewidth = 0.5, alpha = 0.1, drawstyle = 'steps-mid')
            plt.setp(ax2.get_xticklabels(), rotation_mode="anchor")
            
                      
            ax3.set_xticks(np.arange(env.standby.shape[1]))
            ax3.set_yticks(np.arange(env.standby.shape[0]))
            ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')
            
            plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")
            
            fig.suptitle('Final State')
            
            ax1.title.set_text('Render')
            ax2.title.set_text('Visited')
            ax3.title.set_text('Importance Map')
            
            clear_frames = True     # Should it clear the figure between each frame?
            fps = 15
            GifWriter = animation.writers['pillow']
            metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
            writer = GifWriter(fps=fps, metadata=metadata)
            draw={'fig':fig,'ax1':ax1,'ax2':ax2,'ax3':ax3,'simulate_one_case':True,'writer':writer,'video_file':video_file,'video_folder':video_folder}
        else:
            draw={'simulate_one_case':False}
           
          
        reward=run_episode(N,env,model,draw)
      
        metrics = env.metrics()
        
        all_metrics.append(metrics)
      
        
        print("Ended with reward {}".format(reward))
        print("Coverage: {}".format(metrics['coverage']))
        print("Media de tiempo: {}".format(metrics['mean']))
        print("Dev. tipica: {}".format(metrics['std']))
        
        
        if simulate_one_case:
     
            VM = obs['visited_map']
            IM = obs['importance_map']
            img = env.render()
            
            ax1.imshow(img) 
            ax2.imshow(VM, cmap = 'gray')
            im = ax3.imshow(IM,interpolation='bicubic', cmap = 'jet_r')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            plt.show()
            
             
            fig = plt.figure()
            
            plt.imshow(metrics['time_matrix'])
            plt.colorbar()
            
            plt.show()
            plt.suptitle('Mean time waiting in cell')
    
    if simulate_one_case==False:
        cover=[]
        timemean=[]
        for metrics in all_metrics:
            cover.append(metrics['coverage'])
            timemean.append(metrics['mean'])

        covers.append(cover)
        timemeans.append(timemean)
    

if simulate_one_case==False :
    df_cov=pd.DataFrame(covers)
    df_cov=df_cov.transpose()
    df_cov.columns=identifiers
    plot_barblox(df_cov,'Coverage Comparison','Coverage (%)')
             
    df_time=pd.DataFrame(timemeans)
    df_time=df_time.transpose()
    df_time.columns=identifiers
    
    plot_barblox(df_time,'Meantime Comparison','Mean Time (s)')
 
    
