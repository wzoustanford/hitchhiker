from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    token=""  # Need to request access
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Access PyTorch modules directly
print(model)  # Shows all modules
print(model.model.layers[0])  # Access specific transformer layers

# Load custom weights if needed
# model.load_state_dict(torch.load("path/to/checkpoint.pt"))

# Set pad token if not set
tokenizer.pad_token = tokenizer.eos_token

stock_info_prompt = "Ticker is AAPL, company name is Apple Inc., exchange is NASDAQ, country is USA." 

stock_prices_prompt = """
stock prices in the past year, about 248 days: 
[185.6400, 184.2500, 181.9100, 181.1800, 185.5600, 185.1400, 186.1900,
        185.5900, 185.9200, 183.6300, 182.6800, 188.6300, 191.5600, 193.8900,
        195.1800, 194.5000, 194.1700, 192.4200, 191.7300, 188.0400, 184.4000,
        186.8600, 185.8500, 187.6800, 189.3000, 189.4100, 188.3200, 188.8500,
        187.1500, 185.0400, 184.1500, 183.8600, 182.3100, 181.5600, 182.3200,
        184.3700, 182.5200, 181.1600, 182.6300, 181.4200, 180.7500, 179.6600,
        175.1000, 170.1200, 169.1200, 169.0000, 170.7300, 172.7500, 173.2300,
        171.1300, 173.0000, 172.6200, 173.7200, 176.0800, 178.6700, 171.3700,
        172.2800, 170.8500, 169.7100, 173.3100, 171.4800, 170.0300, 168.8400,
        169.6500, 168.8200, 169.5800, 168.4500, 169.6700, 167.7800, 175.0400,
        176.5500, 172.6900, 169.3800, 168.0000, 167.0400, 165.0000, 165.8400,
        166.9000, 169.0200, 169.8900, 169.3000, 173.5000, 170.3300, 169.3000,
        173.0300, 183.3800, 181.7100, 182.4000, 182.7400, 184.5700, 183.0500,
        186.2800, 187.4300, 189.7200, 189.8400, 189.8700, 191.0400, 192.3500,
        190.9000, 186.8800, 189.9800, 189.9900, 190.2900, 191.2900, 192.2500,
        194.0300, 194.3500, 195.8700, 194.4800, 196.8900, 193.1200, 207.1500,
        213.0700, 214.2400, 212.4900, 216.6700, 214.2900, 209.6800, 207.4900,
        208.1400, 209.0700, 213.2500, 214.1000, 210.6200, 216.7500, 220.2700,
        221.5500, 226.3400, 227.8200, 228.6800, 232.9800, 227.5700, 230.5400,
        234.4000, 234.8200, 228.8800, 224.1800, 224.3100, 223.9600, 225.0100,
        218.5400, 217.4900, 217.9600, 218.2400, 218.8000, 222.0800, 218.3600,
        219.8600, 209.2700, 207.2300, 209.8200, 213.3100, 216.2400, 217.5300,
        221.2700, 221.7200, 224.7200, 226.0500, 225.8900, 226.5100, 226.4000,
        224.5300, 226.8400, 227.1800, 228.0300, 226.4900, 229.7900, 229.0000,
        222.7700, 220.8500, 222.3800, 220.8200, 220.9100, 220.1100, 222.6600,
        222.7700, 222.5000, 216.3200, 216.7900, 220.6900, 228.8700, 228.2000,
        226.4700, 227.3700, 226.3700, 227.5200, 227.7900, 233.0000, 226.2100,
        226.7800, 225.6700, 226.8000, 221.6900, 225.7700, 229.5400, 229.0400,
        227.5500, 231.3000, 233.8500, 231.7800, 232.1500, 235.0000, 236.4800,
        235.8600, 230.7600, 230.5700, 231.4100, 233.4000, 233.6700, 230.1000,
        225.9100, 222.9100, 222.0100, 223.4500, 222.7200, 227.4800, 226.9600,
        224.2300, 224.2300, 225.1200, 228.2200, 225.0000, 228.0200, 228.2800,
        229.0000, 228.5200, 229.8700, 232.8700, 235.0600, 234.9300, 237.3300,
        239.5900, 242.6500, 243.0100, 243.0400, 242.8400, 246.7500, 247.7700,
        246.4900, 247.9600, 248.1300, 251.0400, 253.4800, 248.0500, 249.7900,
        254.4900, 255.2700, 258.2000][185.6400, 184.2500, 181.9100, 181.1800, 185.5600, 185.1400, 186.1900,
        185.5900, 185.9200, 183.6300, 182.6800, 188.6300, 191.5600, 193.8900,
        195.1800, 194.5000, 194.1700, 192.4200, 191.7300, 188.0400, 184.4000,
        186.8600, 185.8500, 187.6800, 189.3000, 189.4100, 188.3200, 188.8500,
        187.1500, 185.0400, 184.1500, 183.8600, 182.3100, 181.5600, 182.3200,
        184.3700, 182.5200, 181.1600, 182.6300, 181.4200, 180.7500, 179.6600,
        175.1000, 170.1200, 169.1200, 169.0000, 170.7300, 172.7500, 173.2300,
        171.1300, 173.0000, 172.6200, 173.7200, 176.0800, 178.6700, 171.3700,
        172.2800, 170.8500, 169.7100, 173.3100, 171.4800, 170.0300, 168.8400,
        169.6500, 168.8200, 169.5800, 168.4500, 169.6700, 167.7800, 175.0400,
        176.5500, 172.6900, 169.3800, 168.0000, 167.0400, 165.0000, 165.8400,
        166.9000, 169.0200, 169.8900, 169.3000, 173.5000, 170.3300, 169.3000,
        173.0300, 183.3800, 181.7100, 182.4000, 182.7400, 184.5700, 183.0500,
        186.2800, 187.4300, 189.7200, 189.8400, 189.8700, 191.0400, 192.3500,
        190.9000, 186.8800, 189.9800, 189.9900, 190.2900, 191.2900, 192.2500,
        194.0300, 194.3500, 195.8700, 194.4800, 196.8900, 193.1200, 207.1500,
        213.0700, 214.2400, 212.4900, 216.6700, 214.2900, 209.6800, 207.4900,
        208.1400, 209.0700, 213.2500, 214.1000, 210.6200, 216.7500, 220.2700,
        221.5500, 226.3400, 227.8200, 228.6800, 232.9800, 227.5700, 230.5400,
        234.4000, 234.8200, 228.8800, 224.1800, 224.3100, 223.9600, 225.0100,
        218.5400, 217.4900, 217.9600, 218.2400, 218.8000, 222.0800, 218.3600,
        219.8600, 209.2700, 207.2300, 209.8200, 213.3100, 216.2400, 217.5300,
        221.2700, 221.7200, 224.7200, 226.0500, 225.8900, 226.5100, 226.4000,
        224.5300, 226.8400, 227.1800, 228.0300, 226.4900, 229.7900, 229.0000,
        222.7700, 220.8500, 222.3800, 220.8200, 220.9100, 220.1100, 222.6600,
        222.7700, 222.5000, 216.3200, 216.7900, 220.6900, 228.8700, 228.2000,
        226.4700, 227.3700, 226.3700, 227.5200, 227.7900, 233.0000, 226.2100,
        226.7800, 225.6700, 226.8000, 221.6900, 225.7700, 229.5400, 229.0400,
        227.5500, 231.3000, 233.8500, 231.7800, 232.1500, 235.0000, 236.4800,
        235.8600, 230.7600, 230.5700, 231.4100, 233.4000, 233.6700, 230.1000,
        225.9100, 222.9100, 222.0100, 223.4500, 222.7200, 227.4800, 226.9600,
        224.2300, 224.2300, 225.1200, 228.2200, 225.0000, 228.0200, 228.2800,
        229.0000, 228.5200, 229.8700, 232.8700, 235.0600, 234.9300, 237.3300,
        239.5900, 242.6500, 243.0100, 243.0400, 242.8400, 246.7500, 247.7700,
        246.4900, 247.9600, 248.1300, 251.0400, 253.4800, 248.0500, 249.7900,
        254.4900, 255.2700, 258.2000]
"""
news_prompt = """
news about this company at the start of the month: 
published date:2024-12-19 07:03:17 title: Monster insider trading alert for Apple stock text: For a brief period in mid-December, Apple (NASDAQ: AAPL) appeared poised to become the world\'s first $4 trillion company before the Fed-induced stock market wipe led its shares to a 2.14% 24-hour crash. published date:2024-12-19 08:15:31 title: Apple Aims to Improve Troubled China Business text: The Apple Inc. (NASDAQ: AAPL) iPhone has done well in China for years. published date:2024-12-19 08:32:28 title: EXCLUSIVE: Are Apple, Tesla 2025\'s Tech Titans Or Bubble Trouble? Kurv CEO Dishes On AI, Cloud And More text: "Investors should proceed with cautious optimism," Howard Chan, founder and CEO of Kurv Investment Management, told Benzinga in an exclusive interview. published date:2024-12-19 08:55:47 title: Apple is exploring a way to bring AI features to iPhones in China, report says text: Apple is looking to integrate AI in iPhones sold in China via local partnerships, Reuters reported. Regulatory barriers in China mean Apple is required to partner with domestic AI companies. published date:2024-12-19 09:11:30 title: Apple Reportedly Pitches Chinese Tech Giants on AI Project text: Apple\xa0is reportedly seeking ways to add AI to its smartphones sold in ChatGPT-less China. The iPhone maker is in discussions with two Chinese tech giants —\xa0Tencent\xa0and TikTok parent\xa0ByteDance\xa0— about integrating their artificial intelligence (AI) models into its products, Reuters\xa0reported\xa0Thursday (Dec. 19), citing three sources familiar with the matter. published date:2024-12-19 10:08:46 title: Apple complains Meta requests risk privacy in spat over EU efforts to widen access to iPhone tech text: Apple complained that requests from Meta Platforms for access to its operating software threaten user privacy, in a spat fueled by the European Union\'s intensifying efforts to get the iPhone maker to open up to products from tech rivals. published date:2024-12-19 11:13:15 title: Apple Faces New EU Call to Open iPhone to Competitors text: Apple\xa0is facing new pressure in Europe to let competitors access its iPhone operating system. The European Commission (EC) on Wednesday (Dec. 18)\xa0published\xa0a pair of\xa0documents\xa0covering how the tech giant needs to meet interoperability requirements of the Europe Union\'s (EU)\xa0Digital Markets Act\xa0(DMA). published date:2024-12-19 13:07:00 title: Apple Reportedly Pulls Plug on iPhone Hardware Subscription Plan text: The news reflects a change in Apple\'s overarching payment strategy. published date:2024-12-19 13:35:00 title: Apple and Meta Are Squabbling Again. This Is Why. text: The companies have been arguing over user privacy since 2021, when the iPhone maker launched App Tracking Transparency. published date:2024-12-19 17:11:14 title: Apple Shuts Down Hardware Subscription Dreams text: Apple is said to have shut down its plans for an iPhone subscription service. Bloomberg\'s Mark Gurman has the details on that story and more. published date:2024-12-19 18:51:22 title: Apple (AAPL) Rises As Market Takes a Dip: Key Facts text: Apple (AAPL) closed the most recent trading day at $249.79, moving +0.7% from the previous trading session. published date:2024-12-20 13:00:33 title: Options Trades in NVDA and AAPL text: Nvidia (NVDA) gets a rare price target cut as the U.S. government asks it to probe how parts were able to get into China during elevated trade restrictions. Meanwhile, Apple (AAPL) is in talks with China companies to expand its A.I. published date:2024-12-21 04:30:00 title: 25% of Warren Buffett-Led Berkshire Hathaway\'s $299 Billion Portfolio Is Invested in Only 1 Stock text: Thanks to his unbelievable track record of compounding capital over many decades as the CEO of Berkshire Hathaway , Warren Buffett is probably the most closely watched investor out there. Everyone can learn a thing or two following his advice and actions. published date:2024-12-21 09:30:00 title: If I Had To Invest $100,000 In A Dividend Growth Portfolio Right Now, Here\'s What I Would Buy text: The methodology uses earnings yield, dividend yield, and 5-year dividend CAGR to score and rank 55 stocks across all 11 sectors. St
ocks with no dividends are excluded, ensuring a focus on dividend growth, momentum, and value. The top stocks by sector are evaluated and then backtested in an equal weight format. published date:2024-12-22 07:00:00 title: Apple\'s App Store Puts Kids a Click Away From a Slew of Inappropriate Apps text: An analysis recommends that Apple, an emerging target for digital-safety advocates, apply an independent review for software age ratings. published date:2024-12-22 09:27:36 title: Apple May Be Considering a Branded TV Amid Smart Home Push text: There aren\'t a lot of details, but this has been a rumor for some time. published date:2024-12-23 01:14:29 title: Apple is reportedly developing a home security product that could compete with Amazon and Google text: Apple is developing smart home locks with face recognition tech. This move aligns with Apple\'s growing interest in the home devices market. published date:2024-12-23 04:27:00 title: Why You Should Not Trade Apple Stock Like You\'re Warren Buffett text: Despite a rapidly changing tech landscape, few investors are likely to dispute the assertion that Apple (AAPL 1.88%) stock will remain a long-term winner. While its artificial intelligence (AI) capabilities may not receive as much attention as those of Nvidia or Palantir, its massive resources, improving technology, and customer loyalty will likely keep the company a major force in its industry for some time to come. published date:2024-12-23 05:00:00 title: History Says the Nasdaq Could See Strong Gains in 2025: Here Are the Top Nasdaq Stocks From Warren Buffett and Bill Ackman Going Into the New Year text: The stock market is so hot that I\'m struggling to wrap my mind around the numbers. As of this writing, the Nasdaq Composite Index (^IXIC 1.03%) is up 30% in 2024. published date:2024-12-23 06:57:03 title: Holiday cheer for Apple, says tech-focused investment bank text: Apple Inc\'s (NASDAQ:AAPL, ETR:APC) iPhone 16, launched in September, is showing demand that is “slightly ahead to generally in line with expectations globally,” according to Wedbush analysts. The investment firm predicts a robust holiday season for the technology giant, driven by strong upgrade trends within its extensive customer base.
"""

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

outputs = model.generate(
    **inputs,
    max_new_tokens=2560,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract only the assistant's response
assistant_response = response.split("assistant")[-1].strip()
print(assistant_response)