import tensorflow as tf
import numpy as np

# Define the bandits
bandits = [0.2, 0, -0.2, -5]
number_bandits = len(bandits)
def pullBandit(bandit):
    # get a random number
    result = np.random.randn(1)
    if result > bandit:
        # return a positive reward
        return 1
    else:
        return -1
        
tf.reset_default_graph()
# These two lines defined the feed forward part of the network
weights = tf.Variable(tf.ones([number_bandits]))
chosen_action = tf.argmax(weights, 0)

# Training procedure
reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# Take action, receive reward, update network in order to choose action yeild the highest reward more often


total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(number_bandits) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # Choose either a random action or one from our network
        if np.random.rand(1) < e:
            action = np.random.randint(number_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])
        
        # Update the network
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(number_bandits) + " bandits: " + str(total_reward))
        i+=1
print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
    
    
"""
Running reward for the 4 bandits: [-1.  0.  0.  0.]
Running reward for the 4 bandits: [-1. -3. 15.  0.]
Running reward for the 4 bandits: [-1. -3. 23.  0.]
Running reward for the 4 bandits: [-1. -2. 36.  0.]
Running reward for the 4 bandits: [-5. -2. 40.  0.]
Running reward for the 4 bandits: [-5. -2. 56.  0.]
Running reward for the 4 bandits: [-7. -2. 70.  2.]
Running reward for the 4 bandits: [-6. -1. 68.  4.]
Running reward for the 4 bandits: [-7. -3. 82.  5.]
Running reward for the 4 bandits: [-4. -4. 93.  6.]
Running reward for the 4 bandits: [ -5.  -2. 100.   6.]
Running reward for the 4 bandits: [ -4.  -2. 111.   8.]
Running reward for the 4 bandits: [ -5.  -1. 117.   8.]
Running reward for the 4 bandits: [ -5.   0. 128.   8.]
Running reward for the 4 bandits: [ -5.   0. 136.   8.]
Running reward for the 4 bandits: [ -4.   0. 133.   8.]
Running reward for the 4 bandits: [ -4.   1. 154.   8.]
Running reward for the 4 bandits: [ -5.   1. 172.  11.]
Running reward for the 4 bandits: [ -6.   1. 183.  11.]
Running reward for the 4 bandits: [ -6.   2. 202.  11.]
The agent thinks bandit 3 is the most promising....
...and it was wrong!
"""


"""
Running reward for the 4 bandits: [1. 0. 0. 0.]
Running reward for the 4 bandits: [-1.  1.  9.  0.]
Running reward for the 4 bandits: [1. 0. 4. 2.]
Running reward for the 4 bandits: [-1.  0.  3. 37.]
Running reward for the 4 bandits: [-2. -3.  3. 81.]
Running reward for the 4 bandits: [ -5.  -4.   2. 126.]
Running reward for the 4 bandits: [ -3.  -2.   3. 171.]
Running reward for the 4 bandits: [ -3.  -4.   3. 219.]
Running reward for the 4 bandits: [ -4.  -4.   3. 266.]
Running reward for the 4 bandits: [ -3.  -5.   2. 311.]
Running reward for the 4 bandits: [ -4.  -2.   2. 357.]
Running reward for the 4 bandits: [ -4.  -3.   1. 405.]
Running reward for the 4 bandits: [ -4.  -5.   1. 451.]
Running reward for the 4 bandits: [ -4.  -5.   0. 496.]
Running reward for the 4 bandits: [ -5.  -6.   1. 541.]
Running reward for the 4 bandits: [ -7.  -6.   0. 588.]
Running reward for the 4 bandits: [ -8.  -6.  -1. 636.]
Running reward for the 4 bandits: [ -9.  -6.   0. 684.]
Running reward for the 4 bandits: [ -9.  -5.   1. 732.]
Running reward for the 4 bandits: [-10.  -5.   2. 778.]
The agent thinks bandit 4 is the most promising....
...and it was right!
"""