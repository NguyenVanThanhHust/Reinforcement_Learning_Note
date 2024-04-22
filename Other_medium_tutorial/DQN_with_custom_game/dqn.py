import os
import torch
import torch.nn as nn

class PrioritizedReplayBuffer:
    def __init__(self, capacity, epsilon=1e-6, alpha=0.2, beta=0.4, beta_increment=0.001) -> None:
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
         

        
class DQN(nn.Module):
    def __init__(self, state_shape, action_size, learning_rate_max=0.001, learning_rate_decay=0.995, gamma=0.75, 
                 memory_size=2000, batch_size=32, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.state_shape = state_shape
        self.state_tensor_shape = (-1,) + state_shape
        self.action_size = action_size
        self.learning_rate_max = learning_rate_max
        self.learning_rate = learning_rate_max
        self.learning_rate_decay = learning_rate_decay
        self.gamma = gamma
        self.memory_size = memory_size
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay