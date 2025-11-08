# hitchhiker 

Web server:
cd ~/www/
python3  -m http.server 8080


source venv/bin/activate
python3 -m web.main


Working repository - RL with enrivonment feedback alignment 

- replay buffer is constructed as the training goes on, limited to seen data 
- first collect sampled data from the language model, sample temperature from. [0.7-eps, 0.7+eps] 

What is the RL framework we adopt? 
is it Q learning or direct optimization (PG) 
if it is PG: 

RB: [
    [C_i], # ranked list 
]

What is the action space, should we do any limitation and confinement? The problem with selecting random stocks in a large set is it may confuse the LLM. The problem with asking about individual stocks, while taking actions in a larger set. One of the key insights in search ranking is pair-wise ranking, instead of ranking across the entire list, we can perform random samples of two stocks and do a large amount of comparison training. 

Infra and engineering design, when we implement, we need to consider a limit on the prompt or input length, otherwise, the computlation cost will be too high. In fact, the ROI for each batch optimization may be low. This is because LLM with long context may have limited ability to synthesize all information of the world state, while a comparison analysis not only limits the world state space the LLM needs to synthesize, it also limits the action space of the policy function that the LLM implicitly implements. The comparison analysis instructed by the prompt makes the it easy for the model to obtain meaningful gradients to learn. Finally, the prompt instruction is itself a type of hierarchical RL, and further algorithm and infra could be implemented their to either incorporate expert knowledge or addition ML models. 

Elementwise: 
each ranked list would be the following 
[C_i] = 
[   # for B number of samples 
    (
        (T, state[(price_hsit_e1, news_hist_e1)], next_state[...], a, r), xB #equity 1, text and action are sampled B times 
    ),
    (
        (T, state[(price_hsit_e2, news_hist_e2)], next_state[...], a, r), xB #equity 2, text and action are sampled B times 
    ),
    (
        ...
    ), 
    ...
] 

a = (equity_proportions, advisory_text) 
r is normalized for $1.0. of investment across the length of the investment 

Pairwise: 
each ranked list would be the following 
for the pairwise tuples, choose using combinatory sampling to make sure there are no duplicates 
[C_i] = 
[   # for B number of samples 
    (
        ((Te1, Te2), state[(price_hsit_e1, news_hist_e1), (price_hist_e2, news_hist_e2)], next_state[...], a, r), xB #equity 1 vs 2 the text/action is sampled B times 
    ),
    (
        ((Te1, Te3), state[(price_hsit_e1, news_hist_e1), (price_hist_e3, news_hist_e3)], next_state[...], a, r), xB #equity 1 vs 3 the text/action is sampled B times 
    ),
    (
        ...
    ), 
    ...
]

Listwise: [for later] 

Okay now this data could be collected. How do we fine-tune the LLM after the data is there? 

Elementwise: 
We could use the r's without normalization, the are already normalized to $1.0 dollar value. 
In the ranked list just used supervised learning 
https://epichka.com/blog/2025/grpo/ 

Pairwise: 
Same thing, do it in the minibatch 
