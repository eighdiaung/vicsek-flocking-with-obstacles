# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:05:55 2024

@author: eighd
"""

"""
@author: Eighdi Aung

This code implements the original Vicsek model and the Vicsek model with obstacles used for the paper
"Local interactions in active matter are reinforced by spatial heterogeneity"
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

# Define agent class
class agent:
    def __init__(self, x, y, speed, noise_strength):
        self.x = x # initial x position
        self.y = y # initial y position
        self.speed = speed # speed of particles
        self.noise_strength = noise_strength # process noise in averaging of neighbors' headings
        self.angle = np.random.uniform(0, 2 * np.pi) # initial random heading
        
    def get_position(self): # gets the position of an agent
        return self.x, self.y
    
    def get_heading(self):  # gets the heading of an agent
        return self.angle

# Define the Vicsek model class (this is the Vicsek model in free space without obstacles)
class VicsekModel:
    def __init__(self, num_agents, domain_length = 5.0, speed = 0.25, intrad = 0.5, noise_strength = 0.05, obstacles = None ):
        self.num_agents = num_agents
        self.domain_length = domain_length
        self.speed = speed
        self.intrad = intrad
        self.noise_strength = noise_strength
        self.obstacles = obstacles
        self.agents = self.create_agents()
        self.adj_matrix = np.zeros((self.num_agents, self.num_agents))
        self.positions = np.array([agent.get_position() for agent in self.agents])
        self.headings = np.array([agent.angle for agent in self.agents])
        self.prev_positions = np.array([agent.get_position() for agent in self.agents])
        self.prev_headings = np.array([agent.angle for agent in self.agents])
        
    
    def create_agents(self):
        agents = []
        for _ in range(self.num_agents):
            valid_position = False
            while not valid_position:
                x = np.random.uniform(-self.domain_length/2, self.domain_length/2)
                y = np.random.uniform(-self.domain_length/2, self.domain_length/2)
                
                if self.obstacles is not None:
                    if self.obstacles.ndim == 1: # if single obstacle
                        distance = np.sqrt((x - self.obstacles[0])**2 + (y - self.obstacles[1])**2)
                        if distance > self.obstacles[2]:  # Only accept position outside the obstacle's radius
                            valid_position = True
                    else:
                        # For multiple obstacles:
                        distances = np.sqrt((x - self.obstacles[:, 0])**2 + (y - self.obstacles[:, 1])**2)
                        if np.all(distances > self.obstacles[:, 2]):  # Accept position if all distances are greater
                            valid_position = True
                else:
                    # no need for checking if obstacles are not present
                    valid_position = True
            
            # Once a valid position is found, create the agent
            agents.append(agent(x, y, self.speed, self.noise_strength))
        
        return agents
    
    
    def update_adj(self): # 3 transformations required to check across periodic boundaries to compute adjacency matrix
        agent_positions = self.positions
        distance_matrix = squareform(pdist(agent_positions))
        adj_matrix1 = (distance_matrix < self.intrad).astype(int)
        
        agent_positions = self.positions
        agent_positions[:,0] = np.where(agent_positions[:,0] < 0, agent_positions[:,0] + self.domain_length, agent_positions[:,0])
        distance_matrix = squareform(pdist(agent_positions))
        adj_matrix2 = (distance_matrix < self.intrad).astype(int)
        
        agent_positions = self.positions
        agent_positions[:,1] = np.where(agent_positions[:,1] < 0, agent_positions[:,1] + self.domain_length, agent_positions[:,1])
        distance_matrix = squareform(pdist(agent_positions))
        adj_matrix3 = (distance_matrix < self.intrad).astype(int)
        
        agent_positions = self.positions
        agent_positions = np.where(agent_positions < 0, agent_positions + self.domain_length, agent_positions)
        distance_matrix = squareform(pdist(agent_positions))
        adj_matrix4 = (distance_matrix < self.intrad).astype(int)
        
        # this will be the adjacency matrix if no obstacles are in the environment
        self.adj_matrix_free = adj_matrix1 | adj_matrix2 | adj_matrix3 |adj_matrix4
        
        if self.obstacles is None:
            self.adj_matrix = self.adj_matrix_free.astype(int)
            np.fill_diagonal(self.adj_matrix, 1)  # include self as neighbor for heading average
            return
        
        # if obstacles are present, then remove the indices of agents where the line of sight is blocked by an obstacle
        if self.obstacles is not None:
            
            agent_positions = self.positions
            
            # Tranformation for replicating the domain with 9 adjacent squares to check for periodic bc with obstacles
            tr = np.array([[0, 0],
                           [self.domain_length, 0],
                           [0, self.domain_length],
                           [self.domain_length, self.domain_length],
                           [-self.domain_length, 0],
                           [0, -self.domain_length],
                           [-self.domain_length, -self.domain_length],
                           [self.domain_length, -self.domain_length],
                           [-self.domain_length, self.domain_length]])

            # Replicating the obstacle center transformations
            obs_center = self.obstacles[:, :2]  # Obstacles centers
            obs_centertr = np.repeat(obs_center, 9, axis=0) + np.tile(tr, (obs_center.shape[0], 1))
            obs_rad = np.repeat(self.obstacles[:, 2], 9, axis=0)

            # Loop over time steps
            Ao = np.zeros((self.num_agents, self.num_agents), dtype=bool)  # Original adjacency matrix
            Ai = np.zeros((self.num_agents, self.num_agents), dtype=int)   # Transformation index matrix
            
            # Compute the adjacency matrices considering transformations
            for i in range(9):
                Ao = Ao | (np.sqrt((agent_positions[:, 0] - (agent_positions[:, 0] + tr[i, 0]))**2 +
                                 (agent_positions[:, 1] - (agent_positions[:, 1] + tr[i, 1]))**2) < self.intrad)
                Ai[(np.sqrt((agent_positions[:, 0] - (agent_positions[:, 0] + tr[i, 0]))**2 +
                        (agent_positions[:, 1] - (agent_positions[:, 1] + tr[i, 1]))**2) < self.intrad)] = i
            
                # Removing diagonal elements (self connections)
                Aup = Ao.astype(int) - np.diag(np.diag(Ao)).astype(int)
            
                # Generate all unique pairs of agents
                C = np.array(list(itertools.combinations(range(self.num_agents), 2)))
                

                B = np.zeros([C.shape[0],obs_centertr.shape[0]])  # Blind spot checks matrix
            
                indices = np.where(Aup)
            
                # Blind spot checks
                for iN in range(C.shape[0]):
                    agent1 = C[iN, 0]
                    agent2 = C[iN, 1]
                    # Compute distance to obstacle centers
                    dist_to_obs = np.sqrt(np.sum((np.column_stack([agent_positions[agent1,0], agent_positions[agent1,1]]) - obs_centertr) ** 2, axis=1))
                    obs_indices = np.where(dist_to_obs < self.intrad)[0]
                    
                    if len(obs_indices) > 0:
                        for iobs in obs_indices:
                            # Check if the line between agents intersects the obstacle

                            B[indices[0][iN], iobs] = checkintersect(agent_positions[agent1,0], agent_positions[agent1,1],
                                                                                   agent_positions[agent2,0] + tr[Ai[agent1, agent2], 0],
                                                                                   agent_positions[agent2,1] + tr[Ai[agent1, agent2], 1],
                                                                                   obs_centertr[iobs, 0],
                                                                                   obs_centertr[iobs, 1], obs_rad[iobs])

                B = np.all(B == 0, axis=1)  # If no obstacle blocks the line, keep as neighbors
                Alos = squareform(B)
                A = Ao & Alos              # Combine Ao with the blind spot check
                A = A.astype(int)
                np.fill_diagonal(A, 1)     # Ensure each agent is a neighbor to itself since heading averaging requires including oneself
                self.adj_matrix = A

        
    def check_obstacle_collision(self):
        if self.obstacles.ndim == 1: # if there is only 1 obstacle
            check = ( np.sqrt((self.positions[:,0] - self.obstacles[0])**2 + (self.positions[:,1] - self.obstacles[1])**2)  <= self.obstacles[2] ).astype(int)
            self.positions[:,0] = np.where(check==1, self.prev_positions[:,0], self.positions[:,0])
            self.positions[:,1] = np.where(check==1, self.prev_positions[:,1], self.positions[:,1])
            
            turn_choices = np.random.choice([-1, 1], size=self.num_agents, p=[0.5, 0.5])
            self.headings = np.where(check==1, self.prev_headings +  turn_choices*np.pi/2 , self.headings)

        else:
            for i in range(np.size(self.obstacles,0)):
                check = ( np.sqrt((self.positions[:,0] - self.obstacles[i,0])**2 + (self.positions[:,1] - self.obstacles[i,1])**2)  <= self.obstacles[i,2] ).astype(int)
                self.positions[:,0] = np.where(check==1, self.prev_positions[:,0], self.positions[:,0])
                self.positions[:,1] = np.where(check==1, self.prev_positions[:,1], self.positions[:,1])
                
                turn_choices = np.random.choice([-1, 1], size=self.num_agents, p=[0.5, 0.5])
                self.headings = np.where(check==1, self.prev_headings +  turn_choices*np.pi/2 , self.headings)
                
        for i in range(self.num_agents):
            self.agents[i].x = self.positions[i, 0]
            self.agents[i].y = self.positions[i, 1]
            self.agents[i].angle = self.headings[i]

        
    def move_agent(self,agent):
        # Move in the direction of current angle
        agent.x += agent.speed * np.cos(agent.angle)
        agent.y += agent.speed * np.sin(agent.angle)
        
        # mod for periodic boundary conditions
        agent.x = np.mod(agent.x + self.domain_length/2 , self.domain_length) - self.domain_length/2
        agent.y = np.mod(agent.y + self.domain_length/2 , self.domain_length) - self.domain_length/2
        
        self.positions = np.array([agent.get_position() for agent in self.agents])
        self.headings = np.array([agent.get_heading() for agent in self.agents])
        
        
    def evolve(self):  # evolve the system a single timestep
        self.update_adj()
        
        summed_headings = np.dot(self.adj_matrix, self.headings)
        num_neighbors = np.sum(self.adj_matrix, axis=1)
        num_neighbors = np.where(num_neighbors == 0, 1, num_neighbors)
        avg_angles = summed_headings / num_neighbors
        
        # save previous positions and headings before moving the agents to their next location so we can revert back to their previous position and turn if they violate obstacles
        self.prev_positions = np.array([agent.get_position() for agent in self.agents])
        self.prev_headings = np.array([agent.get_heading() for agent in self.agents])
        
        for i in range(self.num_agents):
            self.agents[i].angle = avg_angles[i] + np.random.uniform(-np.pi, np.pi) * self.noise_strength
            
        for agent in self.agents:
            self.move_agent(agent)
        
        if self.obstacles is not None:
            self.check_obstacle_collision()
            
# code below checks the intersection criteria between line segment between any two agents and obstacle  
# xf, yf are the coordinates of the focal agent,
# x, y are the coordinates of the neighbor to check with
# h ,k are the coordinates of center of the circle
# r is the radius of the circle
def checkintersect(xf, yf, x, y, h, k, r):
    
    # Calculate the slope (m) and intercept (c) of the line
    m = (y - yf) / (x - xf)
    c = yf - m * xf
    
    # Coefficients of the quadratic equation for intersection
    A = 1 + m ** 2
    B = 2 * (m * (c - k) - h)
    C = h ** 2 + (c - k) ** 2 - r ** 2
    
    # Discriminant to check if there is an intersection
    discriminant = B ** 2 - 4 * A * C
    
    if discriminant < 0:
        # No intersection
        return 0
    elif discriminant == 0:
        # One intersection point
        x1 = -B / (2 * A)
        y1 = m * x1 + c
        
        # Check if the intersection point is within the line segment bounds
        if min(xf, x) < x1 < max(xf, x) and min(yf, y) < y1 < max(yf, y):
            return 1
        else:
            return 0
    else:
        # Two intersection points
        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        y1 = m * x1 + c
        
        # Check if the intersection point is within the line segment bounds
        if min(xf, x) < x1 < max(xf, x) and min(yf, y) < y1 < max(yf, y):
            return 1
        else:
            return 0