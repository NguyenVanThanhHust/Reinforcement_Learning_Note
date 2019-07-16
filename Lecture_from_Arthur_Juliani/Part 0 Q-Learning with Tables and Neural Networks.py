##Part 0: Q-Learning with Tables and Neural Networks

"""
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
We will implement Q-learning for Frozen lake
"""

import gym
import numpy as np

# Create Frozen Lake 
env = gym.make('FrozenLake-v0')

#Implement Q Table learning algorithm
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr =  0.8
gamma = 0.95
number_episodes = 2000
# Create list to contain total rewards and steps per episode
rList = []
for i in range(number_episodes):
    # Reset enviroment and get first new obsevation:
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily with noise picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
        # Get new state and reward from enviroment
        s1, r, d, _ = env.step(a) 
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + gamma * np.max(Q[s1, :]) - Q[s,a])
        rAll += rAll
        s = s1
        if d == True:
            break
    rList.append(rAll)
    

print("Score over time: " +  str(sum(rList)/number_episodes)) 
print("Q-Table shape", Q.shape)     
print("Final Q-Table Values")
print(Q * 10)