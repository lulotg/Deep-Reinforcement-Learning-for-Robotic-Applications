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
        
        
        if env.whichrender<1:
            self.state_size = (np.shape(env.map)[0],np.shape(env.map)[1],4)
        elif env.whichrender<3:
            self.state_size = (np.shape(env.map)[0],np.shape(env.map)[1],3)
        else:
            raise('Not valid model')
            
        self.action_size = 8
        self.N=env.N
        self.optimizer = optimizer
        self.loss_function = keras.losses.Huber()
        self.target_episode_update_freq = param_agent['freqUpdate'] #0, is a DQN, >0 DDQN
        # Hyperparameter for RL.
        self.discount_rate = param_agent['gamma'] # Weight of future reward in comparison to current.
        self.epsilon = param_agent['epsilon_max']
        self.epsilon_min = 1 - self.epsilon
        self.epsilon_step = param_agent['epsilon_step']
        # Declaration of the Neural Networks.
        self.sel_model=param_agent['sel_model']
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
            net = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), activation='elu')(input)
            net = Flatten()(net)
            net = Dense(512, activation='elu')(net)
            net = Dense(512, activation='elu')(net)
            net = Dense(512, activation='elu')(net)
            out=[]
            for i in range(self.N):
                out.append(Dense(self.action_size, activation='linear')(net))  
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
            out=[]
            for i in range(self.N):
                out.append(Dense(self.action_size, activation='linear')(net))  
            model = Model(inputs=input, outputs=out)
            model.compile(loss = 'Huber', optimizer = self.optimizer)
        elif self.sel_model==2:  #Action critic propose
            
            print('=====================')    
            print('Original Dueling')
            print('=====================')   
            K=keras.backend
            net = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), activation='elu')(input)
            net = Flatten()(net)
            net = Dense(512, activation='elu')(net)
            net = Dense(512, activation='elu')(net)
            net = Dense(512, activation='elu')(net)
            V = Dense(1)(net) #It returns a single value
            Qs=[]
            for i in range(self.N):
                raw_advantages = Dense(8)(net) #We have the eight states
                #In the following we can put K.max or K.mean
                advantages= raw_advantages - K.mean(raw_advantages, axis = 1, keepdims = True)
                Q= V + advantages
                Qs.append(Q)
            model = Model(inputs=input, outputs=Qs)
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
        actions=[]
        q_values = self.model.predict(state[np.newaxis])
        for i in range(self.N):
            if np.random.rand() < self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                # Exploit/take action with highest chance of reward.
                actions.append(np.argmax(q_values[i]))
        return actions


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
        #HAY QUE DIVIDIRLO AQUI
        if self.target_episode_update_freq>0:  
            #target_Q_values = rewards + self.discount_rate * self.target.predict(next_states)[0][np.argmax(self.model.predict(next_states)[0])]
            Q=self.target.predict(next_states)
        else:           
            Q=self.model.predict(next_states)

        loss=0
        rewards=rewards.transpose()
        actions=actions.transpose()
        all_masks=[]
        all_target_Q_values=[]
        for i in range(self.N):               
            target_Q_values = rewards[i] + self.discount_rate * np.max(Q[i])
            # Reshape the values.
            target_Q_values = target_Q_values.reshape(-1, 1)
            all_target_Q_values.append(target_Q_values)
            # Create a 'one hot' mask.
            all_masks.append(tf.one_hot(actions[i], self.action_size))
            #Calculates the Q of the maximal actions
        with tf.GradientTape() as tape: # Computes the gradient feedforwarding.
            all_Q_values = self.model(states)
            for i in range(self.N): 
                Q_values = tf.reduce_sum(all_Q_values[i] * all_masks[i], axis=1, keepdims=True)
                #Obtains the mean over the batch
                loss = self.loss_function(all_target_Q_values[i], Q_values)+loss
            loss=tf.reduce_mean(loss)
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
