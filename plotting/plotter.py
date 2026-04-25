import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from the CSV file
# Ensure your CSV file is named 'data.csv' and is in the same directory as this script.
try:
    df = pd.read_csv('n2.csv')
except FileNotFoundError:
    print("Error: Please save your data in a file named 'data.csv' first.")
    exit()

# 2. Set up the plot dimensions
plt.figure(figsize=(10, 6))

# 3. Define custom colors for each line to easily distinguish them
colors = {
    'neopt': 'red',
    'opt_m': 'blue',
    'blas': 'green',
    'n2_opt_m': 'darkblue',
    'n2_blas': 'darkgreen'

}

# 4. Loop through each test type, filter the data, and plot the line
for test_type in df['Test'].unique():
    # Get only the rows for the current test type (e.g., just 'neopt')
    subset = df[df['Test'] == test_type]
    
    # Plot X (N) vs Y (Time)
    plt.plot(
        subset['N'], 
        subset['Time'], 
        marker='o', # Adds dots at each data point
        linewidth=2,
        label=test_type, 
        color=colors.get(test_type, 'black') # defaults to black if test_type isn't in dict
    )

# 5. Add labels, title, and scales
plt.title('Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Number of Elements (N)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)

# 6. Add a grid and a legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Test Version', fontsize=10)

# Make the layout tight so nothing gets cut off
plt.tight_layout()

output_filename = 'plot_n2.svg'
plt.savefig(output_filename, format='svg', dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as '{output_filename}' in the current directory.")

# 7. Display the graph
plt.show()
