"""
Combined script for parsing LLEF evaluation results and plotting them.
This combines parse_llama_responses.py and plot_llef_parsed_evaluation_results.py

Usage:
    1. Set OPENAI_API_KEY environment variable
    2. Run: python parse_and_plot_llef_results.py --input <pkl_file> --output <output_prefix>

Example:
    OPENAI_API_KEY='your-key' python parse_and_plot_llef_results.py \
        --input evaluate_llef_first_result_large_cap_cnt151.pkl \
        --output parsed_llef_results_151
"""

import pickle
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI


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


def parse_responses(input_file, output_prefix, api_key):
    """Parse all responses using OpenAI API"""

    # Load the pickle file
    print("Loading pickle file...")
    with open(input_file, 'rb') as f:
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

    # Set up OpenAI client
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
        return None

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
    output_pkl = f'{output_prefix}.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {output_pkl}")

    # Save as JSON for easy inspection
    output_json = f'{output_prefix}.json'
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

    return output_pkl


def plot_results(parsed_file, output_plot='scatter_plot_m_vs_expected.png'):
    """Plot the parsed results"""

    # Load the parsed results
    print("\n" + "="*80)
    print("PLOTTING RESULTS")
    print("n" + "="*80)
    print("Loading parsed results...")
    with open(parsed_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} samples")

    # Extract m values and expected (b) values
    m_values = []
    untrained_expected = []
    trained_expected = []
    tickers = []
    ticker_indices = []
    test_returns = []

    for result in results:
        m = result['m']
        untrained_b = result['untrained_parsed'].get('expected')
        trained_b = result['trained_parsed'].get('expected')

        m_values.append(m)
        untrained_expected.append(untrained_b)
        trained_expected.append(trained_b)

        # Add new fields if they exist
        if 'ticker' in result:
            tickers.append(result['ticker'])
            ticker_indices.append(result['ticker_idx'])
            test_returns.append(result['test_approx_return'])
        else:
            tickers.append(None)
            ticker_indices.append(None)
            test_returns.append(None)

    # Convert to numpy arrays for easier handling (numeric values only)
    m_values = np.array(m_values)
    untrained_expected = np.array(untrained_expected)
    trained_expected = np.array(trained_expected)

    # Keep tickers as list (strings don't need numpy array conversion)
    # Convert ticker_indices and test_returns with object dtype to handle None values
    ticker_indices = np.array(ticker_indices, dtype=object)
    test_returns = np.array(test_returns, dtype=object)

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Filter out None values for untrained
    mask_untrained = untrained_expected != None
    m_filtered_untrained = m_values[mask_untrained]
    untrained_filtered = untrained_expected[mask_untrained].astype(float)

    # Filter out None values for trained
    mask_trained = trained_expected != None
    m_filtered_trained = m_values[mask_trained]
    trained_filtered = trained_expected[mask_trained].astype(float)

    # Filter out None values for test_returns (if available)
    mask_test = test_returns != None
    m_filtered_test = m_values[mask_test]
    test_filtered = []
    for val in test_returns[mask_test]:
        if val is not None:
            test_filtered.append(float(val))
        else:
            # This shouldn't happen due to mask, but just in case
            pass

    has_test_data = len(test_filtered) > 0 and test_filtered[0] != None

    # Plot untrained: (m, untrained_expected) in blue
    ax.scatter(m_filtered_untrained, untrained_filtered, alpha=0.6, s=50, color='blue', label='Untrained Expected (b)', edgecolors='darkblue', linewidth=0.5)

    # Plot trained: (m, trained_expected) in red
    ax.scatter(m_filtered_trained, trained_filtered, alpha=0.6, s=50, color='red', label='Trained Expected (b)', edgecolors='darkred', linewidth=0.5)

    # Plot test_return: (m, test_approx_return) in orange (if available)
    if has_test_data:
        # Need to ensure m_filtered_test and test_filtered have same length
        if len(test_filtered) == len(m_filtered_test):
            test_filtered_array = np.array(test_filtered)
            ax.scatter(m_filtered_test, test_filtered_array, alpha=0.6, s=50, color='orange', label='Actual Test Return', edgecolors='darkorange', linewidth=0.5)

    # Add diagonal line for reference
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='y=x', linewidth=1.5)
    
    # Set axis limits
    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0.9, 1.2)

    ax.set_xlabel('Historical Mean (m)', fontsize=14)
    ax.set_ylabel('Predicted Expected Return (b)', fontsize=14)
    ax.set_title('Historical Mean vs Predicted Expected Return', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_plot}")

    # Calculate correlations
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Untrained model:")
    print(f"  - Successfully parsed: {len(untrained_filtered)}/{len(results)}")
    if len(untrained_filtered) > 0:
        corr_untrained = np.corrcoef(m_filtered_untrained, untrained_filtered)[0, 1]
        print(f"  - Correlation (m vs b): {corr_untrained:.4f}")

    print(f"\nTrained model:")
    print(f"  - Successfully parsed: {len(trained_filtered)}/{len(results)}")
    if len(trained_filtered) > 0:
        corr_trained = np.corrcoef(m_filtered_trained, trained_filtered)[0, 1]
        print(f"  - Correlation (m vs b): {corr_trained:.4f}")

    # Show info about new fields if they exist
    if len(results) > 0 and 'ticker' in results[0]:
        print(f"\nExtended data available:")
        print(f"  - Tickers: {len([t for t in tickers if t is not None])} available")
        print(f"  - Test returns: {len([t for t in test_returns if t is not None])} available")

        # Show correlation with actual test returns if available
        if has_test_data and len(test_filtered) > 0:
            print(f"\nActual Test Returns:")
            print(f"  - Successfully loaded: {len(test_filtered)}/{len(results)}")

            # Calculate correlation between predictions and actual returns
            # Need to align the data points
            test_indices = np.where(mask_test)[0]

            # For untrained predictions vs actual
            untrained_test_aligned = []
            actual_test_aligned = []
            for idx, test_val in zip(test_indices, test_filtered):
                if untrained_expected[idx] is not None:
                    untrained_test_aligned.append(float(untrained_expected[idx]))
                    actual_test_aligned.append(test_val)

            if len(untrained_test_aligned) > 1:
                corr_untrained_actual = np.corrcoef(untrained_test_aligned, actual_test_aligned)[0, 1]
                print(f"  - Correlation (untrained b vs actual): {corr_untrained_actual:.4f}")

            # For trained predictions vs actual
            trained_test_aligned = []
            actual_test_aligned_trained = []
            for idx, test_val in zip(test_indices, test_filtered):
                if trained_expected[idx] is not None:
                    trained_test_aligned.append(float(trained_expected[idx]))
                    actual_test_aligned_trained.append(test_val)

            if len(trained_test_aligned) > 1:
                corr_trained_actual = np.corrcoef(trained_test_aligned, actual_test_aligned_trained)[0, 1]
                print(f"  - Correlation (trained b vs actual): {corr_trained_actual:.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Parse and plot LLEF evaluation results')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file')
    parser.add_argument('--output', type=str, required=True, help='Output prefix for parsed results')
    parser.add_argument('--plot', type=str, default='scatter_plot_m_vs_expected.png', help='Output plot filename')
    parser.add_argument('--skip-parsing', action='store_true', help='Skip parsing and only plot (assumes parsed file exists)')

    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')

    if not args.skip_parsing:
        if not api_key:
            print("\n" + "="*80)
            print("ERROR: OPENAI_API_KEY environment variable not set!")
            print("\nPlease set it with:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            print("\nOr run this script with:")
            print(f"  OPENAI_API_KEY='your-key' python {__file__} --input {args.input} --output {args.output}")
            print("="*80)
            return

        # Parse responses
        parsed_file = parse_responses(args.input, args.output, api_key)

        if parsed_file is None:
            print("Parsing cancelled by user.")
            return
    else:
        parsed_file = f'{args.output}.pkl'
        if not os.path.exists(parsed_file):
            print(f"ERROR: Parsed file {parsed_file} does not exist!")
            print("Remove --skip-parsing flag to parse the data first.")
            return

    # Plot results
    plot_results(parsed_file, args.plot)
    print(f"\nDone! Results saved to {args.output}.pkl, {args.output}.json, and {args.plot}")


if __name__ == "__main__":
    main()
