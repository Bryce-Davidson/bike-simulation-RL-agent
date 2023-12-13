# Reinforcement Learning Overview

The goal of this project is to explore the double Q-learning algorithm, as cited by Van Hasselt et al. [2015], as a method for solving Markov decision processes. Our objective is to optimize a cyclist's power output for a race, as inspired by Anna Kiesenhofer's mathematical approach, which won her the 2021 Tokyo Olympics women's road race. We use the concepts of critical power (CP) and Anaerobic Work Capacity (AWC or W') as our physiological constraints and simple physics equations derived from Newton's Second Law of Motion as our mechanical constraints.

Reinforcement learning is a type of machine learning that models a learning algorithm as an agent. This agent seeks to learn about an environment by taking actions based on its current state, receiving a new state and a reward from that action, and then repeating this process to generate the training data necessary to find a policy for the agent which maximizes its expected cumulative discounted reward over all states. Solving for the maximum discounted reward over all states is commonly referred to as solving the Bellmen equation Sutton and Barto [2018]. In the context of our study, the agent is represented by a cyclist, while the environment encompasses the physiological and mechanical constraints imposed on that cyclist.

## Methods and Materials

## Definitions

- **State space ($S$)**: The set of all possible states.
- **Initial state ($s_0$)**: An initial state in $S$.
- **Action set ($A$)**: The set of actions for which $A(s) \subseteq A$ are the applicable actions our agent can take given it is at state $s \in S$.
- **Transition probability ($P$)**: The transition probability $P_a({s}'\mid s)$ for $s \in S$ and $a \in A(s)$.
- **Reward ($r$)**: A positive or negative reward $r(s,a,{s}')$ given as a result of transitioning from state $s$ to state ${s}'$ using action $a$.
- **Discount factor ($\gamma$)**: A discount factor taking values in the set $[0,1)$ that determines how much a future reward should be discounted compared to a current reward.

## Objective of Reinforcement Algorithms

The objective of reinforcement algorithms is to take an MDP as input, and produce an optimal policy function $\pi_*$ as output, which tells an agent the best action to choose in any given state in order to maximise the expected discounted cumulative reward from that state. The expected value of following a policy $\pi$ from state s, can be represented as a function $V^\pi(s)$ defined as follows,

$$V^\pi(s) = E_\pi[\sum_i \gamma^ir(s_i,a_i,s_{i+1})\mid s_0=s, a_i = \pi(s)]$$

In order to ensure the policy $\pi$ is the optimal policy $\pi_*$, the function $V(s)$ must satisfy the Bellman Equation.

$$V(s) = \underset{a\in A(s)}{max} \sum_{{s}' \in S}P_a({s}' \mid s)[r(s,a,{s}')+\gamma V({s}')]$$

Subsequently, the optimal policy can be extracted by choosing actions which satisfy the following,

$$\pi_*(s) = {argmax}_{a\in A(s)} \sum_{{s}' \in S}P_a({s}' \mid s)[r(s,a,{s}')+\gamma V({s}')]$$

We can simplify this expression by defining a function which represents the value of choosing action $a$ in state $s$ and then following the policy $\pi$.

$$Q(s,a) = \sum_{{s}' \in S}P_a({s}' \mid s)[r(s,a,{s}')+\gamma V({s}')]$$

The optimal policy can then be reformulated as the following which will be the foundation of the Q-learning method,

$$\pi_*(s) = {argmax}_{a\in A(s)} Q(s,a)$$

## Building the Environment

The environment was built around the equations and constants referenced in the appendix [\ref{appendix:Constants}, \ref{appendix:Power Equation}, \ref{appendix:Fatigue Equation}, \ref{appendix:Resistance Equation}, \ref{appendix:Velocity Equation}], to mimic the physiological constraints due to fatigue and recovery, and mechanical constraints due to Newton's Laws of Motion. Additionally, We assumed the rider to have a max AWC of 9758 J, a Critical power of 234 W and mass of 70kg.

Following the implementation of the environment, our next challenge was to devise a reward function that would guide our agent's learning process. Given that our agent's objective is to complete the race as swiftly as possible, we initially set a negative existence reward for each step taken, coupled with a substantial positive reward upon course completion. Furthermore, we aimed to prevent scenarios where the velocity drops below zero, as such situations are undesirable in any racing context. Therefore, we assigned a significant negative reward whenever the current state exhibited a velocity less than zero.

The next step was to train the agent we used the Double Deep Q Networks method which is described in the following section.
