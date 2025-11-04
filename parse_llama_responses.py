import pickle
import json
import os
from openai import OpenAI

# Load the pickle file
print("Loading pickle file...")
with open('evaluate_llef_first_result_large_cap_cnt151.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Extract the results list from the dictionary
data = data_dict['results']
print(f"Loaded {len(data)} samples")

# Detect format
if len(data[0]) == 7:
    print(f"Each sample structure: (untrained_response, trained_response, m, s, ticker, ticker_idx, test_approx_return)")
elif len(data[0]) == 6:
    print(f"Each sample structure: (untrained_response, trained_response, m, s, ticker, ticker_idx) - using default test_approx_return=1.0")
else:
    print(f"Each sample structure: (untrained_response, trained_response, m, s)")

# Show first few responses to understand the noise
print("\n" + "="*80)
print("SAMPLE RESPONSES (first 3):")
print("="*80)
for i in range(min(3, len(data))):
    # Handle both old format (4 elements) and new format (6-7 elements)
    if len(data[i]) == 7:
        untrained_response, trained_response, m, s, ticker, ticker_idx, test_approx_return = data[i]
        print(f"\n--- Sample {i} ---")
        print(f"Ticker: {ticker}, Ticker Index: {ticker_idx}")
        print(f"m (mean): {m}, s (std): {s}, test_approx_return: {test_approx_return}")
    elif len(data[i]) == 6:
        untrained_response, trained_response, m, s, ticker, ticker_idx = data[i]
        test_approx_return = 1.0
        print(f"\n--- Sample {i} ---")
        print(f"Ticker: {ticker}, Ticker Index: {ticker_idx}")
        print(f"m (mean): {m}, s (std): {s}, test_approx_return: {test_approx_return} (default)")
    else:
        untrained_response, trained_response, m, s = data[i]
        print(f"\n--- Sample {i} ---")
        print(f"m (mean): {m}, s (std): {s}")

    print(f"\nUntrained response:\n{untrained_response}")
    print(f"\nTrained response:\n{trained_response}")
    print("-" * 80)

def parse_llama_response_with_gpt(llama_response, client):
    """
    Use OpenAI GPT to parse the noisy llama-3.2 response and extract a, b, c values.

    Returns:
        dict: {"low": a, "expected": b, "high": c} or None values if not found
    """

    parsing_prompt = f"""The following text is a response from a financial forecasting model (llama-3.2-3B-Instruct) that was asked to predict next month's return for a stock.

The model was instructed to provide:
- a%: There is a 1-in-10 chance the actual return will be less than this (10th percentile)
- b%: The expected return (most important)
- c%: There is a 1-in-10 chance the actual return will be greater than this (90th percentile)

The model was asked to return: {{"low": a%, "expected": b%, "high": c%}}

However, the response is noisy and may not be in perfect JSON format. Please intelligently parse the response and extract these three values.

RESPONSE TO PARSE:
{llama_response}

Return ONLY a valid JSON object with numeric values (without % signs):
{{"low": <number>, "expected": <number>, "high": <number>}}

If you cannot confidently extract a value, use null. The "expected" value (b) is the most important."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for better parsing
            messages=[
                {"role": "system", "content": "You are an expert at parsing and extracting structured data from noisy text outputs. Return only valid JSON."},
                {"role": "user", "content": parsing_prompt}
            ],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()

        # Extract JSON from the response (in case there's extra text)
        # Try to find JSON object in the response
        if '{' in result_text and '}' in result_text:
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            json_str = result_text[start:end]
            parsed = json.loads(json_str)
            return parsed
        else:
            return {"low": None, "expected": None, "high": None}

    except Exception as e:
        print(f"Error parsing: {e}")
        return {"low": None, "expected": None, "high": None}


# Main processing
if __name__ == "__main__":
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("\n" + "="*80)
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr run this script with:")
        print("  OPENAI_API_KEY='your-key' python parse_llama_responses.py")
        print("="*80)
        exit(1)

    client = OpenAI(api_key=api_key)

    # Test on first sample
    print("\n" + "="*80)
    print("TESTING PARSER ON FIRST SAMPLE")
    print("="*80)

    # Handle both old format (4 elements) and new format (6-7 elements)
    if len(data[0]) == 7:
        untrained_response, trained_response, m, s, ticker, ticker_idx, test_approx_return = data[0]
        print(f"Ticker: {ticker}, Ticker Index: {ticker_idx}")
        print(f"m: {m}, s: {s}, test_approx_return: {test_approx_return}")
    elif len(data[0]) == 6:
        untrained_response, trained_response, m, s, ticker, ticker_idx = data[0]
        test_approx_return = 1.0
        print(f"Ticker: {ticker}, Ticker Index: {ticker_idx}")
        print(f"m: {m}, s: {s}, test_approx_return: {test_approx_return} (default)")
    else:
        untrained_response, trained_response, m, s = data[0]
        print(f"m: {m}, s: {s}")

    print("\nParsing UNTRAINED response...")
    untrained_parsed = parse_llama_response_with_gpt(untrained_response, client)
    print(f"Result: {json.dumps(untrained_parsed, indent=2)}")

    print("\nParsing TRAINED response...")
    trained_parsed = parse_llama_response_with_gpt(trained_response, client)
    print(f"Result: {json.dumps(trained_parsed, indent=2)}")

    # Ask user if they want to continue
    print("\n" + "="*80)
    user_input = input(f"Process all {len(data)} samples? This will make {len(data)*2} API calls. (y/n): ")

    if user_input.lower() != 'y':
        print("Exiting without processing all samples.")
        exit(0)

    # Process all samples
    print(f"\nProcessing all {len(data)} samples...")
    results = []

    for i, sample in enumerate(data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(data)}")

        # Handle both old format (4 elements) and new format (6-7 elements)
        if len(sample) == 7:
            untrained_resp, trained_resp, m, s, ticker, ticker_idx, test_approx_return = sample
            has_extended_data = True
        elif len(sample) == 6:
            untrained_resp, trained_resp, m, s, ticker, ticker_idx = sample
            test_approx_return = 1.0
            has_extended_data = True
        else:
            untrained_resp, trained_resp, m, s = sample
            has_extended_data = False

        untrained_parsed = parse_llama_response_with_gpt(untrained_resp, client)
        trained_parsed = parse_llama_response_with_gpt(trained_resp, client)

        result_dict = {
            'm': float(m.item()) if hasattr(m, 'item') else float(m),
            's': float(s.item()) if hasattr(s, 'item') else float(s),
            'untrained_response': untrained_resp,
            'trained_response': trained_resp,
            'untrained_parsed': untrained_parsed,
            'trained_parsed': trained_parsed
        }

        # Add extended data if available
        if has_extended_data:
            result_dict['ticker'] = ticker
            result_dict['ticker_idx'] = int(ticker_idx)
            result_dict['test_approx_return'] = float(test_approx_return.item()) if hasattr(test_approx_return, 'item') else float(test_approx_return)

        results.append(result_dict)

    # Save results
    print("\nSaving results...")

    # Save as pickle
    output_pkl = 'parsed_llef_results_151.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {output_pkl}")

    # Save as JSON for easy inspection
    output_json = 'parsed_llef_results_151.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_json}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    untrained_b_values = [r['untrained_parsed']['expected'] for r in results if r['untrained_parsed']['expected'] is not None]
    trained_b_values = [r['trained_parsed']['expected'] for r in results if r['trained_parsed']['expected'] is not None]

    print(f"Untrained: Successfully parsed 'b' (expected) for {len(untrained_b_values)}/{len(data)} samples")
    print(f"Trained: Successfully parsed 'b' (expected) for {len(trained_b_values)}/{len(data)} samples")

    if untrained_b_values:
        print(f"\nUntrained 'b' stats: mean={sum(untrained_b_values)/len(untrained_b_values):.4f}")
    if trained_b_values:
        print(f"Trained 'b' stats: mean={sum(trained_b_values)/len(trained_b_values):.4f}")
