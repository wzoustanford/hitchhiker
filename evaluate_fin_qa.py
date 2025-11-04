from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

# Step 1: Load dataset
print("Loading dataset...")
ds = load_dataset("virattt/financial-qa-10K")
qa_data = ds['train']  # Using train split for evaluation
print(f"Loaded {len(qa_data)} examples")

# Step 2: Load models and tokenizer
print("\nLoading models...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
tokenizer.pad_token = tokenizer.eos_token

# Load untrained model
print("Loading untrained model...")
untrained_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="",
)

# Load trained model
print("Loading trained model...")
trained_model = AutoModelForCausalLM.from_pretrained(
    './checkpoints/step_74500_epoch_5',  # Update this path as needed
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="",
)

# Disable gradients for evaluation
torch.set_grad_enabled(False)

# Step 3: Define get_log_probs function
def get_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities for response tokens only.
    Returns the sum of log probabilities for the answer part.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )

    logits = outputs.logits

    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = response_mask[..., 1:].contiguous()

    # Calculate log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Apply response mask - only count response tokens
    selected_log_probs = selected_log_probs * shift_mask

    # Sum log probs over sequence length
    return selected_log_probs.sum(dim=1)


# Step 4: Process data function
def process_item(item, model, tokenizer):
    """
    Process a single dataset item:
    1. Create prompt from question + context
    2. Tokenize prompt and full sequence (prompt + answer)
    3. Create response_mask
    4. Compute log probs for the answer
    """
    # Step 4a: Create messages with system prompt, context, and question
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant and an expert with US equities."},
        {"role": "user", "content": f"Context: {item['context']}\n\nQuestion: {item['question']}"}
    ]

    # Step 4b: Apply chat template to create prompt (without answer)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant header
    )

    # Step 4c: Create full text (prompt + answer)
    full_text = prompt + item['answer']

    # Step 4d: Tokenize prompt only to get prompt length
    prompt_tokens = tokenizer(
        prompt,
        return_tensors='pt',
        add_special_tokens=False  # Template already has special tokens
    )
    prompt_length = prompt_tokens['input_ids'].shape[1]

    # Step 4e: Tokenize full text (prompt + answer)
    full_tokens = tokenizer(
        full_text,
        return_tensors='pt',
        add_special_tokens=False
    )

    # Step 4f: Create response_mask (0 for prompt, 1 for answer)
    response_mask = torch.zeros_like(full_tokens['input_ids'])
    response_mask[:, prompt_length:] = 1  # Mark answer tokens as 1

    # Step 4g: Move to model device
    input_ids = full_tokens['input_ids'].to(model.device)
    attention_mask = full_tokens['attention_mask'].to(model.device)
    response_mask = response_mask.to(model.device)

    # Step 4h: Compute log probs
    log_prob = get_log_probs(model, input_ids, attention_mask, response_mask)

    return {
        'log_prob': log_prob.item(),
        'prob': torch.exp(log_prob).item(),  # Convert log prob to probability
        'prompt_length': prompt_length,
        'answer_length': full_tokens['input_ids'].shape[1] - prompt_length,
        'question': item['question'],
        'answer': item['answer'],
        'context': item['context'],
        'ticker': item['ticker']
    }


# Step 5: Evaluation loop
print("\n" + "="*80)
print("Starting evaluation...")
print("="*80)

results = []
MAX_ITEMS = 3000  # Start with 100 items to test speed

import time
start_time = time.time()

for idx in range(min(MAX_ITEMS, len(qa_data))):
    item = qa_data[idx]

    # Process with trained model
    print(f"\nProcessing item {idx+1}/{MAX_ITEMS}")
    print(f"Ticker: {item['ticker']}, Question: {item['question'][:80]}...")

    item_start = time.time()

    trained_result = process_item(item, trained_model, tokenizer)
    trained_result['model'] = 'trained'

    untrained_result = process_item(item, untrained_model, tokenizer)
    untrained_result['model'] = 'untrained'

    item_time = time.time() - item_start

    print(f"  Trained log_prob: {trained_result['log_prob']:.4f}, prob: {trained_result['prob']:.6e}")
    print(f"  Untrained log_prob: {untrained_result['log_prob']:.4f}, prob: {untrained_result['prob']:.6e}")
    print(f"  Time: {item_time:.2f}s")

    results.append({
        'idx': idx,
        'trained': trained_result,
        'untrained': untrained_result,
        'processing_time': item_time
    })

    # Print running average time
    if (idx + 1) % 10 == 0:
        avg_time = (time.time() - start_time) / (idx + 1)
        print(f"\n>>> Average time per item: {avg_time:.2f}s")
        print(f">>> Estimated time for {len(qa_data)} items: {avg_time * len(qa_data) / 60:.1f} minutes")

total_time = time.time() - start_time
print("\n" + "="*80)
print(f"Evaluation complete!")
print(f"Total time: {total_time:.2f}s")
print(f"Average time per item: {total_time / len(results):.2f}s")
print("="*80)

# Step 6: Save results
output_file = 'evaluate_fin_qa_results.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {output_file}")

# Step 7: Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

trained_log_probs = [r['trained']['log_prob'] for r in results]
untrained_log_probs = [r['untrained']['log_prob'] for r in results]

print(f"\nTrained model:")
print(f"  Mean log_prob: {sum(trained_log_probs) / len(trained_log_probs):.4f}")
print(f"  Min log_prob: {min(trained_log_probs):.4f}")
print(f"  Max log_prob: {max(trained_log_probs):.4f}")

print(f"\nUntrained model:")
print(f"  Mean log_prob: {sum(untrained_log_probs) / len(untrained_log_probs):.4f}")
print(f"  Min log_prob: {min(untrained_log_probs):.4f}")
print(f"  Max log_prob: {max(untrained_log_probs):.4f}")

log_prob_improvements = [r['trained']['log_prob'] - r['untrained']['log_prob'] for r in results]
print(f"\nLog prob improvement (trained - untrained):")
print(f"  Mean: {sum(log_prob_improvements) / len(log_prob_improvements):.4f}")
print(f"  Items where trained > untrained: {sum(1 for x in log_prob_improvements if x > 0)}/{len(log_prob_improvements)}")
print("="*80)
