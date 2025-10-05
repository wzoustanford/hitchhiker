from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch, re, numpy, pickle 

def parse_eval_score_int(response): 
    try: 
        eval_score_int = int(response.split('[SCORE:')[1].split(']')[0])
    except: 
        try: 
            eval_score_int = response.split('SCORE:')[1].split('\n')[0]
            assert(len(eval_score_int)<=2)
            eval_score_int = int(eval_score_int)
        except: 
            return None 
        return eval_score_int 
    return eval_score_int 

def evaluate_reward(ticker_idx, eval_score_int, D):
    ## function to evaluate the reward using the stock price series in the following X days 

    series = D['train_in_portfolio_series']
    thresh = 5
    if eval_score_int<= thresh:
        return 0
    output = (eval_score_int - thresh) / (thresh * 1.0)
    num_shares = output / (series[ticker_idx, 0] + 1e-10)
    return_series = series[ticker_idx, 1:] * num_shares - series[ticker_idx, 0] * num_shares
    mean_return = torch.mean(return_series)
    stddev = torch.std(return_series)
    sharpe = mean_return / (stddev + 1e-10)
    return sharpe 

batch_size = 5 
filename = 'model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2020_03_25_test_data_start_date_2020_04_26_newsFeaturesTrue_alpacafracfiltered.pkl'
D = pickle.load(open('/home/ubuntu/iclrwzou/finance_data/news_data_25d/'+filename, 'rb'))

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="", 
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Access PyTorch modules directly
print(model)  # Shows all modules
print(model.model.layers[0])  # Access specific transformer layers

# Load custom weights if needed
# model.load_state_dict(torch.load("path/to/checkpoint.pt"))

# Set pad token if not set
tokenizer.pad_token = tokenizer.eos_token
torch.set_grad_enabled(False)
saveL = []

for ticker_idx in range(1000): #range(len(D['trainFeature'])):
    
    ticker = D['all_train_tickers'][ticker_idx]

    stock_info_prompt = f"Ticker is {ticker}."
    
    #print("Processing: " + stock_info_prompt)

    stock_prices_prompt = """
    stock prices in the past year, about 248 days: 
    """ + re.sub(r'\n\s+', ' ', str(D['trainFeature'][ticker_idx]))

    news_prompt = """
    news about this company at the start of the month: 
    """ + str(D['trainNewsFeatures']['strs'][ticker_idx])

    action_request_prompt = """
    From your financial expertise, could you decide on an action to take: 
    How confident are you from 1 to 10, that if we trade this stock, we will make a profit? Please note by 'trade this stock', we mean by buying a unit of this stock tomorrow, holding for 25 days and selling it on the last day. Note for recommendations (R) 1-5, we won't buy the stock, for 6-10, we'll stock proportion to the expression (R-5)/5.0 Please choose a number from 1 to 10 as an ACTION proposal, in the format: [SCORE:R] where R is your ACTION proposal.
    """ 

    messages = [
        {"role": "system", "content": "You are a helpful financial assistant and an expert with US equities."},
        {"role": "user", "content": stock_info_prompt+stock_prices_prompt+news_prompt+action_request_prompt}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    batch_idx = 0
    #for batch_idx in range(batch_size): 
    while batch_idx < batch_size: 
        print('processing batch num: '+str(batch_idx))
        temp = 0.7 + 0.01 * numpy.random.randn()
        with torch.inference_mode():  # Add this context manager
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temp,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        assistant_response = response.split("assistant")[-1].strip()
        #print(assistant_response)

        eval_score_int = parse_eval_score_int(assistant_response)
        if eval_score_int is None: 
            torch.cuda.empty_cache()
            continue
        else: 
            r = evaluate_reward(ticker_idx, eval_score_int, D) 
            rb_tuple = (filename, ticker_idx, batch_idx, ticker, messages, assistant_response, eval_score_int, r)
            print(f"storing ticker_idx:{ticker_idx}, batch_idx:{batch_idx}, ticker:{ticker}, eval_score_int:{eval_score_int}, r:{r}")
            saveL.append(rb_tuple)
            batch_idx += 1 
            torch.cuda.empty_cache()

saveD={'saveL':saveL}
save_filename = 'saveL_fir1k_LLEF_alignment_'+filename 
pickle.dump(saveD, open('/home/ubuntu/iclrwzou/finance_data/llef_alignment_data/' + save_filename, 'wb'))
