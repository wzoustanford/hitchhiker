import pickle, pdb, re 
import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

#filename = 'data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2022_10_09_test_data_start_date_2022_11_10_newsFeaturesTrue_alpacafracfiltered.pkl'
#filename = 'data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2022_11_10_test_data_start_date_2022_12_12_newsFeaturesTrue_alpacafracfiltered.pkl'
filename = 'data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2023_01_13_test_data_start_date_2023_02_14_newsFeaturesTrue_alpacafracfiltered.pkl'
#filename = 'data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2023_02_14_test_data_start_date_2023_03_18_newsFeaturesTrue_alpacafracfiltered.pkl'

D = pickle.load(open(filename, 'rb')) 

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
untrained_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="", 
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

trained_model = AutoModelForCausalLM.from_pretrained(
    './checkpoints/step_74500_epoch_5',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="", 
)

# Set pad token if not set
tokenizer.pad_token = tokenizer.eos_token
torch.set_grad_enabled(False)

large_cap_dict = pickle.load(open('large_cap_filter_dict.pkl', 'rb'))

results = []
MAXCNT = 3000
cnt = 0
print(f'Processing for MAXCNT {MAXCNT} items.... ')
for ticker_idx in range(len(D['trainFeature'])):
    if D['all_train_tickers'][ticker_idx] not in large_cap_dict:
        print(f"--->skiping {D['all_train_tickers'][ticker_idx]}")
        continue
    if cnt > MAXCNT:
        break
    print(f'processing {ticker_idx}th ticker, with cnt:{cnt}')
    prices = D['trainFeature'][ticker_idx]
    monthly_prices = [prices[i].item() for i in range(len(prices)-1, -1, -30)][::-1]
    
    monthly_returns = []
    for i in range(1, len(monthly_prices)):
        monthly_returns.append(monthly_prices[i] / monthly_prices[i - 1])
    
    stock_prices_prompt = """
    Below are the monthly returns for a financial asset over the past 12 months: 
    """ + re.sub(r'\n\s+', ' ', str(monthly_returns))+ """
    Please answer the following questions on next month's return
    There is a 1-in-10 chance the actual return will be less than a%. 
    I expect the next month's return to be: b%.
    There is a 1-in-10 chance the actual return will be greater than c%.
    Please return a JSON object in the following format: 
    '{"low": a%, "expected": b%, "high": c%}'.
    """
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant and an expert with US equities."},
        {"role": "user", "content": stock_prices_prompt}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(untrained_model.device)
    with torch.inference_mode():  # Add this context manager
        outputs = untrained_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    untrained_assistant_response = response.split("assistant")[-1].strip()

    with torch.inference_mode():  # Add this context manager
        outputs = trained_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    trained_assistant_response = response.split("assistant")[-1].strip()

    m = torch.mean(torch.tensor(monthly_returns))
    s = torch.std(torch.tensor(monthly_returns))

    test_last_price = D['train_in_portfolio_series'][ticker_idx][-1]
    test_first_price = D['train_in_portfolio_series'][ticker_idx][0]
    test_approx_return = test_last_price / test_first_price 

    results.append((untrained_assistant_response, trained_assistant_response, m.item(), s.item(), D['all_train_tickers'][ticker_idx], ticker_idx, test_approx_return))
    print(monthly_returns)
    print(stock_prices_prompt)
    print(m)

    cnt += 1

saveD = {'results':results} 

datestr = filename.split('25d_training_data_')[1].split('_newsFeatures')[0]

pickle.dump(saveD, open(f"evaluate_llef_with_test_return_cnt{cnt}_{datestr}.pkl", 'wb'))
