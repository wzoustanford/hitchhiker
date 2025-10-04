from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch, re, numpy

batch_size = 5 
# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    token=""  
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Access PyTorch modules directly
print(model)  # Shows all modules
print(model.model.layers[0])  # Access specific transformer layers

# Load custom weights if needed
# model.load_state_dict(torch.load("path/to/checkpoint.pt"))

# Set pad token if not set
tokenizer.pad_token = tokenizer.eos_token

saveD = {}

for ticker_idx in range(len(D['trainFeature'])):
    stock_info_prompt = f"Ticker is {}, company name is {}., exchange is {}." 

    stock_prices_prompt = """
    stock prices in the past year, about 248 days: 
    """ + re.sub(r'\n\s+', ' ', str(D['trainFeature'][ticker_idx]))

    news_prompt = """
    news about this company at the start of the month: 
    """ + str(D['trainNewsFeatures']['strs'][ticker_idx])

    action_request_prompt = """
    From your financial expertise, could you decide on an action to take: 
    How confident are you from 1 to 10, that if we trade this stock, we will make a profit? Please note by 'trade this stock', we mean by buying a unit of this stock tomorrow, holding for 25 days and selling it on the last day. Please choose a number from 1 to 10 as an ACTION proposal.
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

    for batch_idx in range(batch_size): 
        temp = 0.7 + 0.01 * numpy.random.randn()
        outputs = model.generate(
            **inputs,
            max_new_tokens=2560,
            temperature=temp,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        assistant_response = response.split("assistant")[-1].strip()
        print(assistant_response)

        r = evaluate_reward(response, D) 
    
