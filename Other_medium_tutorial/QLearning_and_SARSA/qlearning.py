import numpy as np
import copy 

# Environemnts 
rewards = np.array([[-float('inf'), -float('inf'), 0, 0, -float('inf'), -float('inf'), -float('inf')],
           [-float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), 0],
           [0, -float('inf'), -float('inf'), 0, -float('inf'), 100, -float('inf')],
           [0, -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf')],
           [-float('inf'), 0, -float('inf'), 0, -float('inf'), 100, 0],
           [-float('inf'), -float('inf'), 0, -float('inf'), 0, 100, 0],
           [-float('inf'), 0, -float('inf'), -float('inf'), 0, 100, -float('inf')]])

# Paramters
gamma = 0.8
alpha = 0.01
num_episode = 50000
min_difference = 1e-3
goal_state = 5

def QLearning(rewards, goal_state=None, gamma=0.99, alpha=0.01, num_episode=1000, min_difference=1e-6):
    """ 
    Run Q-learning loop for num_episode iterations or till difference between Q is below min_difference.
    """
    return 