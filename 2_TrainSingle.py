import numpy as np
from tqdm import tqdm
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import lfilter
import pickle 
import time

# Import defined interactions with the environment.
import EnvironmentSingle as Environment
# Import agent actions and model.
import AgentSingle as Agent


def do_step(env,action):
    
    obs, reward, done, info = env.step(action)   
    state = env.render()
    return state, reward, done, info

def reset(env):
    
    env.reset()
    state=env.render()
    return state


#param_env={'incW':0.05,'decRel':50,'rOutMin':0,'rOutMax':10,'rInMin':-255,'rInMax':255,'penalty':-10,'setRelevanceMap':False,'render':2}
#param_env={'incW':0.055,'decRel':50,'rOutMin':-0.5,'rOutMax':5,'rInMin':-255,'rInMax':255,'penalty':-10,'setRelevanceMap':False,'render':0}
#freqUpdate=0 : DQN, > 0 DDQN, being the value for how many espisodes is done
#param_agent={'freqUpdate':5,'isOriginal':False,'batch_size':250,'num_of_episodes':1000,'steps_per_episodes':250,'mem_size':10000,'gamma':0.95,'alpha':0.001}
#param={'param_agent':param_agent,'param_env':param_env }
file_map='../Data/Maps/BinaryMap.csv'
folder_net='../Data/Single1/Networks/'
folder_train='../Data/Single1/TrainingCurves/'
file = open('../Data/Settings/param_listSingle1.pickle', 'rb')
param_list=pickle.load(file)
file.close()
hmtrain=len(param_list)

print('There are '+str(hmtrain)+ 'scenarios')

reduce_for_testing=False #Change to train the real scenarios

#You can select here which scenarios to do
for i in range(hmtrain):
   
    param=param_list[i]
    param_env=param['param_env']
    param_agent=param['param_agent']
    
    if reduce_for_testing:
        #Change this values if you want to check that the code is running
        param_agent['batch_size']=10
        param_agent['num_of_episodes']=10
        param_agent['steps_per_episode']=40
    
    

    # Kill all sessions running (for clear start point).
    keras.backend.clear_session()
    # Get the environment (this will get all methods).
    
    
    env = Environment.environment(file_map,param_env)
    # We must reset the environment.
    env.reset()
    
    
    """
        TRAINING PARAMETERS DEFINITION
        Here we will establish the values for our parameters.
    """
    
    alpha=param_agent['alpha']
    batch_size = param_agent['batch_size']
    num_of_episodes = param_agent['num_of_episodes']
    steps_per_episode = param_agent['steps_per_episodes']
    best_score = -1000000 # High number on purpose.
    reward_buffer = []
    filtered_reward_buffer = []
    episode_buffer = []
    filtered_reward = 0
    first = 1
    loss = -1
    
    if param_agent['sel_model']==0:
        stringNet='DQNModelOrig'+str(param_agent['freqUpdate'])
    elif param_agent['sel_model']==1:
        stringNet='DQNModelFer'+str(param_agent['freqUpdate'])
    else:
        stringNet='Dueling'+str(param_agent['freqUpdate'])
    
    optimizer = keras.optimizers.Adam(learning_rate = alpha)
    asv = Agent.agent(env,optimizer,param_agent) # ASV stands for Autonomous Surface Vehicle.
    
    np.random.seed(56)
    tf.random.set_seed(56)
    
    print('Scenario:'+str(i)+' '+stringNet)
    asv.model.summary()
   
    start_time = time.time()
    time_buffer=[]
    print(param)
    print('Render:'+str(env.whichrender)+' Model:'+str(asv.sel_model)+' Freq:' +str(asv.target_episode_update_freq))
    # Loop for the number of episodes.
    for episode in tqdm(range(0,num_of_episodes)):
        # Reset the environment so we start in 'clean' slate on each episode.
        state = reset(env)
        # Set the reward gathered on this episode to zero.
        reward_episode = 0
        # Append the number of episode to buffer for control.
        episode_buffer.append(episode+1)
        # Start execution of episode (take the steps).
        for step in range(steps_per_episode):
            # Update the epsilon value. With this we ensure exploration at the beginning and exploitation at the end.
            asv.epsilon = max(asv.epsilon - asv.epsilon_step, asv.epsilon_min)
            # Take an action.
            action = asv.take_action(state)
            # Get results from action.
            next_state, reward, done, info = do_step(env,action)
            # Save in memory the results.
            asv.append_memory(state,action,reward,next_state,done)
            # Update the state.
            state = next_state
            # Add current reward to total.
            reward_episode += reward
            # If variable 'done' is set to one, we don"t have episodic training.
            if done: break
            # If there is enough data (buffer is bigger than the batch).
            if len(asv.replay_memory) > batch_size:
                # Compute the loss function.
                loss = asv.make_a_training_step(batch_size)
        # Load into the buffer the reward of current episode.
        reward_buffer.append(reward_episode)
        time_buffer.append(time.time()-start_time)
        # If this is the first episode, receive the reward 'unalthered'.
        if first == 1:
            filtered_reward = reward_episode
            first = 0
        else:
            filtered_reward = 0.95*filtered_reward + reward_episode*0.05
        # Append the filtered reward to the buffer.
        filtered_reward_buffer.append(filtered_reward)
        # If reward is greater than the actual best score, update and notify.
        if reward_episode > best_score:
            best_weights = asv.model.get_weights() # Save the weights of the NN.
            best_score = reward_episode
            print("\nEP: " + str(episode) + " --- New high score with reward: " + str(best_score))
        # Update the weights of the target model.
        asv.update_target(episode)
    
    
    """
        When training is done (previous steps), we retrieve the best performing
        NN weights and save it in a file.
    """
    # Save last used and best performing weights.
    strtime=datetime.now().strftime('-%m-%d-%H-%M')
    
    
    if not os.path.isdir(folder_net):
        os.makedirs(folder_net)
    
        
    file_common=str(i)+'_'+stringNet+strtime    
  
    
    asv.model.save(folder_net+'LAST_NN_'+file_common+'.h5')
    asv.model.set_weights(best_weights)
    asv.model.save(folder_net+'BEST_NN_'+file_common+'.h5')
    
    
    # Save the buffers for late retrieval and comparing.
    if not os.path.isdir(folder_train):
        os.makedirs(folder_train)
        
        
    file = open(folder_train+file_common+'.names', 'bw')
    file.close()
        
            
    np.savetxt(folder_train+'rewardBuffer_'+file_common+'.csv', reward_buffer, delimiter=",")
    np.savetxt(folder_train+'episodeBuffer_'+file_common+'.csv', episode_buffer, delimiter=",")
    np.savetxt(folder_train+'filteredRewardBuffer_'+file_common+'.csv', filtered_reward_buffer, delimiter=",")
    np.savetxt(folder_train+'timeBuffer_'+file_common+'.csv', time_buffer, delimiter=",")
   
    
    param_save={'param_agent':param_agent,'param_env':param_env,'param_map':file_map }
    with open(folder_train+'param_'+file_common+'.pickle', 'wb') as handle:
        pickle.dump(param_save, handle)
    
    #np.savetxt(datetime.now().strftime('Training/episodeBuffer_DQN-%m-%d-%H-%M.csv'), episode_buffer, delimiter=",")
    #np.savetxt(datetime.now().strftime('Training/filteredRewardBuffer_DQN-%m-%d-%H-%M.csv'), filtered_reward_buffer, delimiter=",")
    
    # Plot evolution of the reward with the episodes.
    plt.figure(figsize=(12, 6))
    plt.plot(episode_buffer,reward_buffer,'b',alpha=0.2)
    plt.plot(episode_buffer,filtered_reward_buffer,'r')
    plt.grid(True, which = 'both')
    plt.xlim([0,num_of_episodes])
    plt.title("Reward per episode")
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.show()
    
    # Get the standard deviation.
    std = np.sqrt(np.power(np.asarray(filtered_reward_buffer) - np.asarray(reward_buffer),2))
    std = lfilter(np.asarray([0.05]), np.asarray([1,-0.95]), std)
    
    # Plot the rewards and it's std deviation as a function of the episodes.
    plt.figure(figsize=(12, 6))
    plt.plot(episode_buffer,filtered_reward_buffer,color='g', linewidth=1.5)
    plt.fill_between(episode_buffer,filtered_reward_buffer+std,filtered_reward_buffer-std, color = 'green', alpha=0.3)
    plt.grid(True, which = 'both')
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Sum of awards", fontsize=14)
    plt.show()
