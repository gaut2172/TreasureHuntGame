# TreasureHuntGame

TreasureHuntGame is a pathfinding AI project. It is a game where the player needs to find the treasure before the AI pirate agent does. Using a deep Q-learning algorithm, the pirate learns the optimal way to find the treasure from any position on the map.

The algorithm used achieves a 90.5% win rate against the given player in less than 7 minutes. The pirate agent starts out with no knowledge of the environment and within the first 20 games learns how to win, consistently beating the player for the following 180 games.

## Tools Used

This project uses the TensorFlow and Keras libraries in Python to implement deep reinforcement learning and to solve the pathfinding problem.

## Artificial Intelligence Explanation

Machine learning is a subset of AI that uses artificial neural networks. This particular project uses a deep reinforcement learning approach with two neural networks. Reinforcement learning uses the Markov decision process which involves states, actions, and rewards. For example, the NNs make predictions based on past actions in similar game states which granted specific rewards. After processing the environment, the NNs output the Q-value of all possible actions and decide which action grants the highest reward. After executing the chosen action, the NNs update its model weights to remember and improve.

Opposed to Q-learning, deep Q-learning NNs do not store all the possible actions and states in a Q-table. Instead, it uses an approximation function to generalize Q-values, which results in a machine learning approach suited to more complex environments with more possible actions.

## Screenshots

![Game map](Screenshots/map.jpg)
![Problem solved in less than 7 minutes](Screenshots/problemSolved.jpg)

## CS-370 Specific Questions

I developed the code that encompasses the reinforcement learning algorithm. I wrote the game loops that utilize Keras and Tensorflow to achieve the pathfinding solution. I was given the game setup code that includes GameExperience.py and TreasureMaze.py.

Computer scientists solve complex problems with technology in innovative ways. I approach problems with a technical mindset. I use resources at my disposal and current technology knowledge to solve problems efficiently.

One of the most important ethical concerns I learned about in this course regards hidden bias in AI. Any AI system can have biases injected into it and can harm our society without us aware being aware of it. As a developer of machine learning, I need to be aware of my own biases and attempt to design systems that are not inherently biased.
