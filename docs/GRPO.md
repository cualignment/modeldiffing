# Grouped Relative Policy Optimization (GRPO) Summary 

training LLMs for reasoning purposes 
think of GRPO as ranking the answers relationally, rather than simply saying what is right or wrong  
    AKA, tell the model, don't just answer the question, give me a series of steps justifying the answer

    two main ways to train a model to give intermediate steps of reasoning: self-supervised learning and reinforcement learning

    reinforcement learning --> basically a game against itself  
        write a solution using a bunch of steps 
        use the same model to check if the steps are ok one by one
        you get a really strong grader from this model
        creating better and better solutions 

    in self supervised learning, we need people to produce solutions that can be compared to the model output --> but this takes a lot of people, a ton of data, and has a limit to how smart the answer can get by the best answer given by a person (limited --> can only be as good as the top person creating data)

    playing chess for ex: either we can teach the computer lots of different algorithms to play a game, or we can teach it to play against itself  

# situating GRPO and PPO — big picture steps to language model learning and where GRPO and PPO fit in 

Training a language model generally follows 3 major phases: pretraining, supervised fine tuning, and reinforcement learning from human feedback. 

Pretraining: aims to learn general language patterns by predicting the next word. Uses massive amount of data across the internet to enact unsupervised leanring (usually next-token prediction via causal language modeling). Creates a base LLM that can generate fluent text, but lacks human intent/alignment. 

Supervised fine tuning: aims to teach the model how to follow instructions or behave in a useful way, using data that maps a prompt to an ideal response curated by humans. Creates a model that starts doing what humans want, answer questions, write helpful code, etc, --> but still not realiably aligned with human prefs. 

    In short, explicit answers are compared to the LLMs answers, "this is what you should say when asked this." Correct answer. 

Reinforcement learning from human feedback: aims to align the model more closely with human values. Generates multiple completions for a prompt, asks humans or a reward model to rank or score them. Done through reinforcement learning, usually GRPO or PPO. Creates a finetuned LLM that better matches human judgement. 

    In short, rankings or scores, "this is how you should feel about your answers." Possibly two correct answers, but one is better than the other. Matters most for subjective, open ended, or safety critical tasks. --> what answers feel aligned with human values? 


# Where GRPO fits in



# Introduction

Core reinforcement learning algorithm behind DeepSeek. The reason why they were (sort of) able to train their reasoning model at a fraction of the cost of other models incured by openAI. 

What limitation of PPO or other RL methods is GRPO responding to? 
GRPO is a successor of PPO. 

What is the "grouped" idea in GRPO? 

What makes GRPO different from PPO (mathematically or conceptually)? 
Its basically just a tweak to OpenAI's PPO, build on reinforcement learning research.

# Summary

Randomly initialized model that outputs gibberish. Then, we pretrain it by predicting the next token on the entirety of the internet. But still not fully usuable as an assistant because it just finishes sentence and builds on top of it. To make it more conversational, we run instruction fine tuning using instruction response pairs. 

To make further improvements, reinforcement learning comes in. Helps train models with explicit reference answers and uses implicit signs instead (think of chatGPT asking "Which response do you prefer? Please select"). --> called preference finetuning --> multiple implementations --> Reinforcement Learning with Human Feedback (RLHF) which requires training an additional rewards model based on human preference (PPO). A more recent and computationally efficient alternative to RLHF is Direct Preference Optimization (DPO) --> open source community has embraced DPO.

Special class of problems called reasoning tasks, which includes math, CS, logic, etc. There is usually one right answer and a deterministic way of finding whether the model was correct. Reasoning tasks may be trained through Reinforcement Learning with Verifiable Rewards (RLVR) --> updates the model weights based on correctness signal -- using openAI's PPO or deepseek's GRPO (more efficient alternative).

# Reinforcement learning

At each timestep t, the agent takes an action a(t), which moves it to a new state, s(t). The environment can also communicate a reward, r(t), to reflect how much progress the machine has made. The trajectory tau consists of all the states and actions visited and performed during one episode/cycle. 
The agent is an LLM like lama 
The environment is the world external to the LLM --> humans, data sets, tools like python interpreter, etc.

 We can only judge the response quality at the very end of the response. Compare the final answer against the ground truth. If the model is right, it will get a positive reward.

big picture: agent takes a serious of actions and navigates a series of states, recieivng a reward at the end of the episode. 

But somehow, based on this single numeric value, we need to back propagate through entire trajectory and update the weights to reinforce good actions (INCREDIBLY DIFFICULT)

agent uses its policy to produce a probability distribution over the next actions given the current state. This is saying, we are passing the instruction and the tokens generated so far to the LLM, which gives us a probability for each next token in the vocab. The token with highest probability ends up being sampled. If we get positive reward for this, we know to increase the weight of highest prob token for that location. 

Policy Gradient Methods (both PPO and GRPO):
Increase the probability of a token by following the policy gradient. Simply doing gradient ascent on the policy surface (we want to get closer to the peak of the mountain). Do this by calculating the gradient, and taking a step in the gradient's direction. --> calculation for updating model weights by taking a step in the direction of the gradient (see screenshot).

Notes on the equation:
Sum over all the tokens in the response (this is the same as all the timesteps). Take the log of the policy probability, just bc its mathematically more convenient (even though this still pulls the ball in the direction of the gradient). Multiply the gradient by the reward of the final response. When reward is positive, we go up the mountain, zero we dont make an update, negative, we shrink the probability/go down the mountain. 


VOCAB
* Policy (pi index theta) --> aka the LLM, where theta are the model parameters 
* action a(t)
* state s(t)
* reward r(t)
* trajectory (tau)
