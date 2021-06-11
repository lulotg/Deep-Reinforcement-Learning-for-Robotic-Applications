import numpy as np
from numpy import genfromtxt

class environment:
    """
        Initializing all variables related to the interaction with the environment.
    """
    def __init__(self,map,param_env):
        # Load the binary map created (csv file) previously.
        self.map = genfromtxt(map, delimiter=',',dtype = int)
       
        #Increment idlleness (standby) value
        self.incW=param_env['incW']
        
        #Decrement relevance value
        self.decRel=param_env['decRel']
        
        self.rInMin=param_env['rInMin']
        self.rInMax=param_env['rInMax']
        self.rOutMin=param_env['rOutMin']
        self.rOutMax=param_env['rOutMax']
        self.slope=(self.rOutMax-self.rOutMin)/(self.rInMax-self.rInMin)
        self.penaly=param_env['penalty']
       
        self.setRelevanceMap=param_env['setRelevanceMap']
        self.whichrender=param_env['render']
       
       # Create matrixes for the states.
        
        # Standby (W(i,i)) is for controlling the status of each cell. 
        # If set to zero, it has just been visited.
        # It is incremented incW at any time step where it is not visited.
        # Its maximum value is 1 (at some moment, we have not visited for long enough)
        # It is multiplied by the relevance, to obtan R
        
        self.standby = np.ones(self.map.shape)
            
        # Relevance is for controlling how relevant is to visit/revisit that cell. 
        #This value decreases over time (decRel), in each visit. 
     
        self.relevance = np.ones(self.map.shape)*255
        
        if self.setRelevanceMap:
            index=np.where(self.map==0)
            self.relevance[index]=0
                
        
        # Visited is for controlling which of the cells has been visited. 
        #If set to 255, cell has been just visited.
        #If set to 125, cell was previously visited.
        #It is required, because standby goes from 0 to 1 slowly. 
        #And this value is just to know which cells were visited. 
        self.visited = np.zeros(self.map.shape)
        
        # Matrixes for controling the coverage of vehicles.
        # They go in hand with previous matrixes, only when testing (not required for training)
        #They will help us generate metrics in the future.
        
        #How many times a cell is visited
        self.cells_visited = np.zeros(shape = self.map.shape)
        #For how long, since last visit, the cell is waiting to be visited again
        self.cells_waiting = np.ones(shape = self.map.shape)
        #Accumulated waiting times between visits
        self.cells_time = np.zeros(shape = self.map.shape)
        
        # Defines if it will train or test the environment. Mainly for efficiency, if we are testing, makes no sense to 'retrain' model if we have one already.
        self.test_mode = False
        
        # Randomly start from point (x,y).
        # We got to find a (x,y) that is not zero (actual water, not land).
        initial_x, initial_y = np.nonzero(self.map)
        init_cell_index = np.random.randint(0,initial_x.size)
        # With determined index, set starting position.
        self.agent_start_position = (initial_x[init_cell_index], initial_y[init_cell_index])
        self.agent_position = self.agent_start_position
        self.agent_position_previous = self.agent_start_position
       
        # Update the starting position (mark as visited):
        self.visited[self.agent_position[0]][self.agent_position[1]] = 255
        self.standby[self.agent_position[0]][self.agent_position[1]] = 0
        self.relevance[self.agent_position[0]][self.agent_position[1]] -= self.decRel


    """
        Set all values to initial/zero. Will delete progress.
    """
    def reset(self):
        # Set all state matrixes to starting point.
        # All cells are in standby, all are relevant and all haven't been visited.
        self.standby = np.ones(self.map.shape)
        self.relevance = np.ones(self.map.shape)*255
        self.visited = np.zeros(self.map.shape)
        
        
        if self.setRelevanceMap:
            index=np.where(self.map==0)
            self.relevance[index]=0
                
        
        # Set all covergae matrixes to starting point.
        # All haven't been visited, all are waiting and time hasn't started yet.
        self.cells_visited = np.zeros(shape = self.map.shape)
        self.cells_waiting = np.ones(shape = self.map.shape)
        self.cells_time = np.zeros(shape = self.map.shape)
        # Randomly start from point (x,y)
        initial_x, initial_y = np.nonzero(self.map)
        init_cell_index = np.random.randint(0,initial_x.size)
        # With determined index, set starting position.
        self.agent_start_position = (initial_x[init_cell_index], initial_y[init_cell_index])
        self.agent_position = self.agent_start_position
        self.agent_position_previous = self.agent_start_position
        # Update the starting position (mark as visited):
        self.visited[self.agent_position[0]][self.agent_position[1]] = 255
        self.standby[self.agent_position[0]][self.agent_position[1]] = 0
        self.relevance[self.agent_position[0]][self.agent_position[1]] -= self.decRel

        # Get the current state: it returns the three posibilities required 
        # to define the state of the original paper (position, visited, R)
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.standby*self.relevance
        obs['position'] = self.agent_position
        return obs


   
    """
        Method that takes a step given an action.
        Checks if it's possible, updates the matrixes, computes the reward and returns the state.
    """
    def step(self,action):
        future_position = np.copy(self.agent_position_previous)
        # Decompose between 8 possible actions agent can take.
        if action == 0: # North
             future_position[0] -= 1
        elif action == 1: # South
             future_position[0] += 1
        elif action == 2: # East
             future_position[1] += 1
        elif action == 3: # West
             future_position[1] -= 1
        elif action == 4: # Northeast
             future_position[0] -= 1
             future_position[1] += 1
        elif action == 5: # Northwest
             future_position[0] -= 1
             future_position[1] -= 1
        elif action == 6: # Southeast
             future_position[0] += 1
             future_position[1] += 1
        elif action == 7: # Southwest
             future_position[0] += 1
             future_position[1] -= 1
        # Check if movement/action is possible
        ilegal_movement = 0
        # If position is not water but land.
        if self.map[future_position[0]][future_position[1]] == 0:
            ilegal_movement = 1
        else:
            self.agent_position = future_position
            
        # Update visited map with 2 positions. Shade previous one and highlight current.
        self.visited[self.agent_position_previous[0]][self.agent_position_previous[1]] = 127
        self.visited[self.agent_position[0]][self.agent_position[1]] = 255
        
        # Update previous position's relevance, this way we will indicate that the site has been visited and is not relevant at the moment.
        # By decreasing the relevance, we will force the agent to find other cells that are relevant in order to earn a higher reward.
        self.relevance[self.agent_position_previous[0]][self.agent_position_previous[1]] =  np.max([self.relevance[self.agent_position_previous[0]][self.agent_position_previous[1]]-self.decRel,0])
        
        # Compute the reward for each action.
        # Get how good is next movement by looking at relevance and standby matrixes.
        rho_next = self.relevance[self.agent_position[0]][self.agent_position[1]] * self.standby[self.agent_position[0]][self.agent_position[1]]
        # Get how good is current state by looking at relevance and standby matrixes.
        rho_current = self.relevance[self.agent_position_previous[0]][self.agent_position_previous[1]] * self.standby[self.agent_position_previous[0]][self.agent_position_previous[1]]
        # Determine how good is the action by the difference between previous and future action.
        # Depending on relevance, values of both rho's will be in range [0,255]
        reward = rho_next - rho_current
        # Now, depending on the legality of the movement, we decide how much we want to penalize the actions.
        # If ilegal_movement == 1: reward = -10
        # If ilegal_movement == 0: reward = [0, 10] (will be in range)
        #   If rho_next and rho_current are 0 -> reward = 5
        #   If rho_next is 0 and rho_current is 255 -> reward = 0
        #   If rho_next is 255 and rho_current is 0 -> reward = 10
        #   If rho_next is 255 and rho_current is 255 -> reward = 5      
        #reward = (1-ilegal_movement) * ((5/255)*(reward-255)+10) - ilegal_movement*(10)
        reward = (1-ilegal_movement) * (self.slope*(reward-self.rInMax)+self.rOutMax) + ilegal_movement*(self.penaly)
       
        # Update matrix 'standby'
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                self.standby[i][j] = np.min([self.standby[i][j]+self.incW, 1])
        self.standby[self.agent_position[0]][self.agent_position[1]] = 0
        
        # Update position after action.
        self.agent_position_previous = self.agent_position
        # Get the current state.
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.standby*self.relevance
        obs['position'] = self.agent_position
        
        # Variable that if activated, will end the process.
        done = 0 # Set to zero for end episodic training: we dont need it, because our episodes never end
        
        # When test mode is activated, update the cells visited and the time that passed since last visit.
        if(self.test_mode == True):
            # Increment number of visits to cell
            self.cells_visited[self.agent_position[0]][self.agent_position[1]] += 1
            # Copy the waiting time for the cell #
            wait_time = self.cells_waiting[self.agent_position[0]][self.agent_position[1]]
            # Increment waiting time for all cells #
            self.cells_waiting +=  1
            #Waiting time for visited cell will be 1 next time
            self.cells_waiting[self.agent_position[0]][self.agent_position[1]] = 1
            # Accumulated waiting time for the cell #
            self.cells_time[self.agent_position[0]][self.agent_position[1]] += wait_time
        else: # test mode deactivated.
            pass
        
        
        return obs, reward, done, ilegal_movement


    """
        Render map with color: careful, is also used for training !
    """
    def render0(self):
        land_color = np.asarray([0,160,20])/255 #Green
        water_color = np.asarray([0,0,0])/255 #Cyan
        agent_color = np.asarray([255,0,0])/255 #Yellow
        # First, we copy the map.
        size_map = (self.map.shape[0],self.map.shape[1],3)
        base_map = np.zeros(size_map)
        # Loop through the columns.
        for i in range(0,self.map.shape[0]):
            # Loop through the rows.
            for j in range(0,self.map.shape[1]):
                # Land/ground (in binary map cell values are 0).
                if(self.map[i][j] == 0):
                    base_map[i][j] = land_color
                # If when looping agent is found, paint cell in 'agent_color'.
                elif(self.agent_position[0] == i and self.agent_position[1] == j):
                    base_map[i][j] = agent_color
                # If cell has been visited
                elif(self.visited[i][j] != 0):
                    # Get how relevant is the cell based on its visits.
                    # The most visited/less relevant, the brighter.
                    state_of_visited = (255-self.relevance[i][j]*self.standby[i][j])/255
                    visited_color = [state_of_visited,state_of_visited,state_of_visited]
                    base_map[i][j] = visited_color
                else:
                    base_map[i][j] = water_color
        return base_map
    
    
    
    def render1(self):
        land_color = np.asarray([0,255,0])/255 #Green
        water_color = np.asarray([0,255,255])/255 #Cyan
        agent_color = np.asarray([255,255,0])/255 #Yellow
        # First, we copy the map.
        size_map = (self.map.shape[0],self.map.shape[1],3)
        base_map = np.zeros(size_map)
        # Loop through the columns.
        for i in range(0,self.map.shape[0]):
            # Loop through the rows.
            for j in range(0,self.map.shape[1]):
                # Land/ground (in binary map cell values are 0).
                if(self.map[i][j] == 0):
                    base_map[i][j] = land_color
                # If when looping agent is found, paint cell in 'agent_color'.
                elif(self.agent_position[0] == i and self.agent_position[1] == j):
                    base_map[i][j] = agent_color
                # If cell has been visited
                elif(self.visited[i][j] != 0):
                    # Get how relevant is the cell based on its visits.
                    # The most visited/less relevant, the brighter.
                    state_of_visited = (255-self.relevance[i][j]*self.standby[i][j])/255
                    visited_color = [state_of_visited,state_of_visited,state_of_visited]
                    base_map[i][j] = visited_color
                else:
                    base_map[i][j] = water_color
        return base_map
    
    def render2(self):
        #land_color = np.asarray([0,255,0])/255 #Green
        #water_color = np.asarray([0,255,255])/255 #Cyan
        #agent_color = np.asarray([255,255,0])/255 #Yellow
        # First, we copy the map.
        size_map = (self.map.shape[0],self.map.shape[1],3)
        base_map = np.zeros(size_map)
        # Loop through the columns.
        for i in range(0,self.map.shape[0]):
            # Loop through the rows.
            for j in range(0,self.map.shape[1]):
                # Land/ground (in binary map cell values are 0).
                #base_map[i][j][0]=self.map[i][j]*255
                if(self.agent_position[0] == i and self.agent_position[1] == j):
                    base_map[i][j][1] = 255  #Channel 1 where it is placed
                # If cell has been visited
                if(self.visited[i][j] != 0):
                    # Get how relevant is the cell based on its visits.
                    # The most visited/less relevant, the brighter.
                    state_of_visited = (255-self.relevance[i][j]*self.standby[i][j])
                    base_map[i][j][0] = state_of_visited
                    base_map[i][j][2]=255   #Mapa: celda visitable
                elif (self.map[i][j] != 0): 
                    base_map[i][j][2]=125   #Celda visitable, pero no visitada todavia
                    
        base_map=base_map/255
        return base_map


    def render(self):
        if self.whichrender==0:
            base_map=self.render0()
        elif self.whichrender==1:
            base_map=self.render1()
        else:
            base_map= self.render2()
        return base_map
        
    

    def set_test_mode(self,test_mode = False):
        self.test_mode = test_mode
        if(test_mode == True):
            print("Test mode is enabled. To disable it, go to 'Environment_actions.py' in Line 25 and set to 'False'")
        else:
            print("Test mode is disabled. To enable it, go to 'Environment_actions.py' in Line 25 and set to 'True'")


    """
        Determine, compute and get metrics for retrieval.
    """
    def metrics(self):
        if self.test_mode == False:
            print("There are no metrics to return. Test mode is disabled. To enable it, go to 'Environment_actions.py' in Line 25 and set to 'True'")
            return -1
        # Coverage metric.
        number_cells_covered = np.count_nonzero(self.visited)
        number_cells_tovisit = np.count_nonzero(self.map)
  
        coverage = number_cells_covered/number_cells_tovisit
        # Mean frequency of visits.
        visit_freq = []
        
        index=np.where(self.cells_visited==0)
        # Get the time a 'spot' has to wait to be visited.
        aux=self.cells_visited
        aux[index]=1
        wait_time_visit= self.cells_time/aux
        # Loop through the rows.
        for i in range(0,self.map.shape[0]):
            # Loop through the columns.
            for j in range(0,self.map.shape[1]):
                if(self.map[i][j] == 1 and wait_time_visit[i][j] != np.inf and wait_time_visit[i][j] > 0):
                    visit_freq.append(wait_time_visit[i][j])
        # Determine the mean and standard deviation.
        mean = np.mean(visit_freq)
        std = np.std(visit_freq)
        # Create and fill metrics dictionary for returning information.
        metrics_dict = {}
        metrics_dict['coverage'] = coverage
        metrics_dict['mean'] = mean
        metrics_dict['std'] = std
        metrics_dict['time_matrix'] = wait_time_visit
        metrics_dict['visited_acc_matrix'] = self.cells_visited
        metrics_dict['visit_freq'] = visit_freq
        return metrics_dict
