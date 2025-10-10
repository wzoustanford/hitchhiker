from transformers import pipeline
import torch 

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a funny friend and always makes jokes with your tech bros!"},
    {"role": "user", "content": "try to say this in a funny way: How is life and how is research? Are you working hard? Don't be cause your friend wants you to be happy and spend more fun time in the sun."},
]
outputs = pipe(
    messages,
    max_new_tokens=2560,
)
print(outputs[0]["generated_text"][-1])