import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the parsed results
print("Loading parsed results...")
with open('parsed_llef_results_151.pkl', 'rb') as f:
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
output_file = 'scatter_plot_m_vs_expected.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_file}")

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

plt.show()
