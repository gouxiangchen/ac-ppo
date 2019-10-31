# ac-ppo
pytorch implementation Actor-Critic and openAI clipped PPO in gym cartpole-v0 and pendulum-v0 environment

# introduction
implement A2C and PPO in pytorch

# requirement 

- tensorflow (for tensorboard logging)
- pytorch (>=1.0, 1.0.1 used in my experiment)
- gym

# a2c

a2c in cartpole and pendulum, the training result shows below

![a2c-cartpole](./a2c-cartpole-result.png)
<center>result of a2c in cartpole-v0</center>

![a2c-pendulum](./a2c-pendulum-result.png)
<center>result of a2c in pendulum-v0, it's quite hard for a2c converge in pendulum.. </center>

# ppo
![ppo-pendulum](./ppo-pendulum-result.png)
<center>result of ppo in pendulum-v0, somehow still hard to converge..don't know why, any one helps?</center>

# ppo improved 
![ppo-modified](./ppo-modified.png)
<center>more efficient update with generalized advantage estimator (GAE) 