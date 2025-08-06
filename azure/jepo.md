1. i would like to add a training recipe named jepo
2. i need the algorithm looks like

Firstly i will introduce how to do jepo function.

consider I have batch of prompt (question), denote as x and answer, denote as a. also a generated rollout.n responses
for each response, first use the delimiter (a string inputted as hyperparamter) to split the response, and treat the first split as chain-of-though, denote as c_j, where j is the index of the response.
also use i as the index of the response, we now calculate A_i=log(\frac{1}{n}\sum_{j=1}^n\pi_\theta(a|x,c_j))-v_i
where v_i=log(\frac{1}{n-1}\sum_{j\neq i}\pi_\theta(a|x,c_j))
calculate A_i for all n response.
calculate tilde_A_i=clip(A_i/std(A),-1,1) where std(A) is the std across all A_i
Then consider the format reward A_i_format=-p, where p is also the hyper param. If delimiter in response, A_i_format is 0, otherwise -p. Then calcualte format reward for all response, and then calculate the normalized A_format which is (A_i_format - mean) / std where mean is the mean of A_i_format for all i and std is same way to calculate std. The normalized format advantage is note as tilde_A_i_ref
the calculate the grad1 as \frac{1}{n}\sum_{i=1}^n((tilde_A_i + tilde_A_i_ref)\nable_\theta log\pi_\theta(c_i|x))
then calculate grad2 as \nable_\theta log(\frac{1}{n}\sum_{i=1}^n \pi_\theta(a|x,c_i))
the use the existing kl divergence loss gradient as \nable_\theta KL(\pi_\theta(.|x),\pi_ref(.|x))
the overall gradient is grad1+beta_suppgrad2-betagrad3


okay. now consider the typical grpo workflow.
1. main a buffer with max length B, if the buffer is full, do jepo with some steps.
2. do typical grpo training, for a question if all response are not correct (reward is 0), do reample as recipe/dapo
3. and for those all incorrect question, record x,a, and n response
4. when the buffer is full, do jepo.

please also give me a bash such that i can test whether it work as expected.