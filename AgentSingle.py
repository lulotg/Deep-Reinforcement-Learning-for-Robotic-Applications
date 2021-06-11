import numpy as np
from collections import deque
import random
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten


class agent:
    """
        Initialization of the agent with all parameters.
    """
    def __init__(self,env,optimizer,param_agent):
        self.state_size = (np.shape(env.map)[0],np.shape(env.map)[1],3)
        self.action_size = 8
        self.optimizer = optimizer
        self.loss_function = keras.losses.Huber()
        self.target_episode_update_freq = param_agent['freqUpdate'] #0, is a DQN, >0 DDQN
        # Hyperparameter for RL.
        self.discount_rate = param_agent['gamma'] # Weight of future reward in comparison to current.
        self.epsilon = param_agent['epsilon_max']
        self.epsilon_min = 1 - self.epsilon
        self.epsilon_step = param_agent['epsilon_step']
        self.sel_model=param_agent['sel_model']
        # Declaration of the Neural Networks.
        self.model, self.target = self.build_model()
        # Memory for storing the experiences.
        self.replay_memory = deque(maxlen=param_agent['mem_size'])
       


    """
        Save memory of previous step. Append it to buffer created.
    """
    def append_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append([state, action, reward, next_state, done])


    """
        Creation of the neural networks (both model and target).
        It's important that the input layer has the same size as the 'state_size'.
        Also, the output layer size should be the same as 'action_size'.
    """
    def build_model(self):
        input = Input(shape=(self.state_size))
        if self.sel_model==0: #Original propose for QLearners
            print('=====================')    
            print('Original Model Q')
            print('=====================')    
            net = Conv2D(8, kernel_size=(5, 5), strides=(2, 2), activation='elu')(input)
            net = Flatten()(net)
            net = Dense(1024, activation='elu')(net)
            net = Dense(1024, activation='elu')(net)
            out = Dense(self.action_size, activation='linear')(net)  
            model = Model(inputs=input, outputs=out)
            model.compile(loss = 'Huber', optimizer = self.optimizer)
        elif self.sel_model==1: #Luis Fernando propose for QLearners (less parameters)
            print('=====================')    
            print('Luis Fernando Model for Q')
            print('=====================')   
            net = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input)
            net = Flatten()(net)
            net = Dense(512, activation='relu')(net)
            net = Dense(512, activation='relu')(net)
            out = Dense(self.action_size, activation='linear')(net)
            model = Model(inputs=input, outputs=out)
            model.compile(loss = 'Huber', optimizer = self.optimizer)
        elif self.sel_model==2:  #Action critic propose
            print('=====================')    
            print('Original Dueling')
            print('=====================')   
            K=keras.backend
            net = Conv2D(8, kernel_size=(5, 5), strides=(2, 2), activation='elu')(input)
            #This could have been flatten, and done something as in the previous case
            net = Flatten()(net)
            net = Dense(512, activation='relu')(net)
            V = Dense(1)(net) #It returns a single value
            raw_advantages = Dense(8)(net) #We have the eight actions
            #In the following we can put K.max or K.mean. The multi paper says mean
            advantages= raw_advantages - K.mean(raw_advantages, axis = 1, keepdims = True)
            Q= V + advantages
            model = Model(inputs=input, outputs=Q)
            model.compile(loss = 'Huber', optimizer = self.optimizer)
        
        
        # Both NN are equal, so we copy the created above.
        target = keras.models.clone_model(model)
        target.set_weights(model.get_weights())
        return model, target

    """
        With epsilon greedy policy, decide which action to take now.
        The value of epsilon will decide how much we explore/exploit.
        The value will be updated while training.
    """
    def take_action(self,state):
        if np.random.rand() < self.epsilon:
            # Explore/take random action.
            return random.randrange(self.action_size)
        else:
            # Exploit/take action with highest chance of reward.
            q_values = self.model.predict(state[np.newaxis])
            return np.argmax(q_values[0])


    """
        Given a batch size, determine the attributes for each.
        From the replay memory, get batch_size random experiences.
        With this batch, retrieve each experience's attributes.
    """
    def sample_experiences(self,batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, done = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, done


    """
        Take a step with a given batch.
        Get a batch of experiences.
        Apply the algorithm (DQN).
        Feedforward the net, compute and apply the gradient.
        Obtain and return the loss.
    """
    def make_a_training_step(self,batch_size):
        # Sample a batch of experiences.
        states, actions, rewards, next_states, done = self.sample_experiences(batch_size)
        #states, actions, rewards, next_states, done = experiences
        # Get the target values. This is DQN algorithm.
    
        if self.target_episode_update_freq>0:  
            #target_Q_values = rewards + self.discount_rate * self.target.predict(next_states)[0][np.argmax(self.model.predict(next_states)[0])]
            target_Q_values = rewards + self.discount_rate * np.max(self.target.predict(next_states),axis=1)
        else:           
            target_Q_values = rewards + self.discount_rate * np.max(self.model.predict(next_states),axis=1)

        # Reshape the values.
        target_Q_values = target_Q_values.reshape(-1, 1)
        # Create a 'one hot' mask.
        mask = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape: # Computes the gradient feedforwarding.
            all_Q_values = self.model(states)
            #Calculates the Q of the maximal actions
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            #Obtains the mean over the batch
            loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
            # Compute the gradient.
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply the gradient.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


    """
        Update target's weights based on the frequency defined.
    """
    def update_target(self,episode):
            if self.target_episode_update_freq>0:
                if((episode % self.target_episode_update_freq) == 0):
                    self.target.set_weights(self.model.get_weights())
                else:
                    pass
            else:
                pass    
