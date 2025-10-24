import pickle, pdb, re 
import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

filename = 'data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2022_10_09_test_data_start_date_2022_11_10_newsFeaturesTrue_alpacafracfiltered.pkl'
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
    './checkpoints/step_15500_epoch_1',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="", 
)

# Set pad token if not set
tokenizer.pad_token = tokenizer.eos_token
torch.set_grad_enabled(False)

large_cap_dict = pickle.load(open('large_cap_filter_dict.pkl', 'rb'))

results = []
MAXCNT = 100
cnt = 0
for ticker_idx in range(len(D['trainFeature'])):
    if D['all_train_tickers'][ticker_idx] not in large_cap_dict:
        print(f"--->skiping {D['all_train_tickers'][ticker_idx]}")
        continue
    if cnt > MAXCNT:
        break
    print(f'processing {ticker_idx}th ticker, with cnt:{cnt}')
    prices = D['trainFeature'][ticker_idx]
    monthly_prices = [prices[i].item() for i in range(len(prices)-1, -1, -30)][::-1]
    
    for i in range(1, len(monthly_prices)):
        monthly_prices[i] = monthly_prices[i] / monthly_prices[i - 1]
    monthly_returns = monthly_prices[1:]
    
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

    results.append((untrained_assistant_response, trained_assistant_response, m, s))
    cnt += 1

saveD = {'results':results} 
pickle.dump(saveD, open(f"evaluate_llef_first_result_large_cap_cnt{cnt}.pkl", 'wb'))
