# Proximal Policy Optimization (PPO)

## Introduction 

PPO is an algorithm in deep reinforcement learning (RL) that aims to improve the efficiency of deep reinforcement learning while maintaining stable behavior. Like other policy gradient methods, PPO represents its decision making process as a policy: a neural network parametrized by theta, (input latex notation here) that outputs the probability of taking action a given state s. 

PPO aims to maximize the agent's cumulative reward by adjusting theta over time, making actions with higher returns more likely. PPO improves this policy using gradient ascent, while a clipping function limits how much the new policy can deviate from the old one.

## From Rewards to Advantage 

After the agent takes action at in state st, the environment provides a reward rt. However, rewards can be noisy or shortsights. To make learning more stable, we compute an advantgae fucntion At, which estimates how much better or worse an action was compared to the expected value of that state. 

This advantage guides policy updates so that actions with **positive advantage** are reinforced, and actions with **negative advantage** are discouraged.

The advantage is over computed using a **value function** Vs, trained alongside the policy. 

{INPUT ONE STEP ESTIMATE AND VALUE FUNCTION HERE}


### Policy Gradient 

The policy itself is parametrized as \( \pi_\theta(a \mid s) \), a neural network that outputs the probability of taking action \( a \) in state \( s \), given weights \( theta \). The goal is to improve the policy by increasing the probability of actions that yield higher long-term rewards. 

PPO uses a policy gradient to update the weights. Policy gradients compute a direction for improving the policy, telling the agent how to adjust its parameters for future decisions. The computation of the policy gradient is as follows: 

\[
\nabla_\theta J(\theta) \approx \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right]
\]
[^1][^2]

Where: 
- \( \nabla_\theta J(\theta) \) is the gradient of the policy's expected reward with respect to its parameters 
- \( \hat{\mathbb{E}}_t [\cdot] \) is an average over time - across many time steps or episode the agent has traversed during training 
- \( \nabla_\theta \log \pi_\theta(a_t \mid s_t) \) measures how sensitive the action probabilities are to the neural network's current parameters 
- \( \hat{A}_t \) is the **advantage estimate**, guiding the size and direction of the update. 


## Clipped Objective Function 

The clipped objective function works to prevent the new policy from deviating too much from the old one. 


## References 

[^1]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/abs/1707.06347  

[^2]: OpenAI Spinning Up. *Proximal Policy Optimization*. https://spinningup.openai.com/en/latest/algorithms/ppo.html
