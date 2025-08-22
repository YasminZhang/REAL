I would like to have the folder structure the same way as when i run grpo when using verl/trainer/main_ppo.py. while i need to make some modification. the modification is mainly about to rewrite a reward function and calculate the necessary component to calculate the reward.

recall that when we run grpo, the naive reward is 0 or 1 when we extract the final answer by matching \\boxed{} from response, and then do string matching with the ground truth. while now i need to calculate the reward for each response by this way.

after doing rollouts.
1. use the inputted delimier (for example \\boxed{), and split each response by the delimiter (noet that all responses are tokens, so please decode them, split, and encode back to cot). the first half we call it cot, denote as c.
2. for this question, get the ground truth from dataset, denote as a. denote the question in the dataset as x.
3. the reward of this response is calculate by the likelihood rario \pi_theta(a|c,x)/\pi_theta(a|x). please pay attention to the reference model and current model used for update.
4. put all custom reward func script and code under recipe/cot_reward folder, and give a bash to run it. the base sctrutcure can refer recipe/dapo/jepo_only_1e5.sh