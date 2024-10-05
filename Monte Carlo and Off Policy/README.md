# Monte Carlo and Off Policy Methods
## On Policy
In on-policy learning the agent follows the same policy for taking actions as well as estimating policy values.

In our sample policy we have 2 actions. One with an 80% probability and one with a 20% probability.

An episode is simulated in our sample through the agent taking aan action based on the policy at each time step.
The agent will then receive a reward based on the action selected. 

The Monte Carlo method here estimates the value function by finding the averaged sum of the returns across multiple episodes.

## Off Policy
In off-policy learning, our goal is to estimate the target policy π. We do this by using
a behaviour policy μ.

In the sample Python file, the target policy π is defined with an action that has an 80%
probability and an action that has a 20% probability.

The behaviour policy has 2 actions that have an equal probability of occurring.

### Importance Sampling
The idea behind importance sampling is to adjust the returns of the behaviour policy by the 
probability ratio of the target policy and behavior policy.

Each step of the learning we compute the importance sampling ratio ρ. The ratio is given by ρ=π/μ.

We then find the weighted return by finding the product of importance weights and total returns across an episode.
The value function is then found by dividing the total of the weighted returns by the total of the importance weights.

## Safe Off Policy Evaluation
The goal of safe off policy evaluation is to provide a lower bound for the target policy. The goal is to provide
confidence that the lower bound of the target policy is >= the lower bound at all times.

The confidence bound is derived using the Chernoff-Hoeffding inequality. 

