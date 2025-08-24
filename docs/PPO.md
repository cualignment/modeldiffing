# Proximal Policy Optimization (PPO)

## Introduction 

PPO is a reinforcement learning (RL) algorithm that helps an agent learn good behavior by trial and error, while keeping learning **stable and efficient**.  

At the heart of PPO is a **policy** — a neural network with parameters $ \theta $ that outputs probabilities for which action to take in a given state $ s $:

$$
\pi_\theta(a \mid s).
$$

Think of it as: *“Given what I see (the state), how likely am I to move left, right, or fire?”*  

The goal of PPO is to **increase the chance of good actions** (actions that lead to higher rewards) while making sure we don’t change the policy too drastically in one step (which could make learning unstable).

---

## From Rewards to Advantage 

When the agent takes an action $ a_t $ in state $ s_t $, it gets a reward $ r_t $.  

But raw rewards are noisy — sometimes you get lucky, sometimes unlucky. So instead of just using the reward, PPO looks at the **advantage**:  

*Was this action better than what I usually expect in this situation?*

The **value function** tells us how good a state is on average:

$$
V^\pi(s_t) = \mathbb{E}_\pi \!\left[ \sum_{k=0}^{\infty} \gamma^k\, r_{t+k} \;\middle|\; s_t \right].
$$

Then we compare the reward to this baseline. A simple version is the **temporal-difference (TD) advantage**:

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

- If $ A_t > 0 $, the action was better than expected → reinforce it.  
- If $ A_t < 0 $, the action was worse than expected → discourage it.  

For stability, PPO often uses **Generalized Advantage Estimation (GAE)**, which blends many time steps to reduce noise:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l},
$$

with  

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

So advantage is essentially the agent asking itself:  
*“Was this move actually better than what I normally do here?”*  

---

## Policy Gradient 

The policy (the agent’s strategy) is a probability distribution $ \pi_\theta(a \mid s) $.  

We want to **nudge** the policy so that actions with positive advantage get more likely, and actions with negative advantage get less likely. This is done with the **policy gradient**:

$$
\nabla_\theta J(\theta) \;\approx\; \hat{\mathbb{E}}_t \Big[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; \hat{A}_t \Big].
$$

- $ \nabla_\theta \log \pi_\theta(a_t \mid s_t) $ says *“how much does the probability of this action depend on my network weights?”*  
- Multiplying by $ \hat{A}_t $ means good actions push weights one way, bad actions push them the other way.  
- Averaging over time smooths things out.  

In plain words: *“Increase the chance of good actions, decrease the chance of bad ones.”*

---

## Clipped Objective Function 

Here’s where PPO innovates. Vanilla policy gradient methods can change the policy too much in one update, breaking learning.  

PPO adds a **clip** to stop updates from being too extreme.  

First, define the ratio of the new policy vs. the old one:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

- If $ r_t(\theta) = 1 $, nothing changed.  
- If it’s much larger or smaller, the policy is changing too fast.  

The clipped objective is:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \Big[
\min\big( r_t(\theta)\, \hat{A}_t,\;
\text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\, \hat{A}_t \big)
\Big].
$$

The idea is simple:  
- If the update is small → let it happen.  
- If the update is too big → squash it back into range.  

This keeps learning steady rather than jumpy.

---

## Full PPO Objective 

In practice, PPO also balances three things:
1. **Clipped policy update** (don’t change too much).
2. **Value function loss** (make state predictions accurate).
3. **Entropy bonus** (encourage exploration by keeping some randomness).

The full loss is:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\!\left[
L^{\text{CLIP}}(\theta)
- c_v \, \big(V_\theta(s_t) - V_{\text{targ},t}\big)^2
+ c_H \, \mathcal{H}\!\big(\pi_\theta(\cdot \mid s_t)\big)
\right],
$$

where:  
- $ c_v $ = weight for value prediction accuracy.  
- $ c_H $ = weight for keeping the policy a bit random (entropy).  
- $ V_{\text{targ},t} $ = the value target used for training.  

In plain words: PPO = *“Improve the policy steadily, keep value estimates accurate, and don’t become too predictable.”*

---

## Intuition Recap

- **Policy ($ \pi $)** = the agent’s strategy (probabilities of actions).  
- **Value ($ V $)** = a baseline for how good a state usually is.  
- **Advantage ($ A $)** = “Was this action better or worse than normal?”  
- **Policy gradient** = push probabilities up for good actions, down for bad ones.  
- **Clip** = keep updates from being too wild → stable learning.  
- **Entropy** = keep some randomness → don’t get stuck.  

PPO is basically:  
*A careful way to nudge the policy in the right direction, without letting it change too drastically.*  

---

## References 

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/abs/1707.06347  

OpenAI Spinning Up. *Proximal Policy Optimization*. https://spinningup.openai.com/en/latest/algorithms/ppo.html  
