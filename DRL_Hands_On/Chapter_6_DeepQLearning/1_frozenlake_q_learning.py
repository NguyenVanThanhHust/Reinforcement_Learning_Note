"""
Tabular Q-Learning algorithm
1. Start with emtpy table for Q(s, a)
2. Obtain (s, a, r, s') from the environment
3. Update with Bellman equation:
Q(s, a) <- (1-alpha)*Q(s,a) + alpha*(reward + gamma*maxQ(s', a'))
4. Check converge condition
"""

import os
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self) -> None:
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)
    
    def sample_env(self):
        """
        Sample random action from action space,
        return (old state, taken action, obtained reward and new state)
        """
        action  = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        if is_done:
            self.state = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, reward, new_state
    
    def best_value_and_action(self, state):
        """
        Receive state of env and find the best action 
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action
    
    def value_update(self, s, a, r, next_s):
        """
        Update value table and calculate Bellman equation
        """
        best_value, _ = self.best_value_and_action(next_s)
        new_value = r + GAMMA * best_value
        old_value = self.values[(s, a)]
        self.values[(s, a)] = old_value*(1-ALPHA) + new_value*ALPHA

    def play_episode(self, env):
        """
        Play one full episode
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _, best_action  = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(best_action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
    

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no +=1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s) 

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /=  TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated: {} -> {}".format(best_reward, reward))
            best_reward = reward
        if reward > 0.9:
            print("Solved in {} iteration".format(iter_no))
            break
    writer.close()
