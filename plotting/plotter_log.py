import pandas as pd
import matplotlib.pyplot as plt

# 1. Încărcarea datelor din fișierul CSV
try:
    df = pd.read_csv('n2.csv')
except FileNotFoundError:
    print("Eroare: Te rog să salvezi datele într-un fișier numit 'data.csv' în același folder.")
    exit()

# 2. Setarea dimensiunilor graficului
plt.figure(figsize=(10, 6))

# 3. Definirea culorilor personalizate pentru fiecare algoritm
colors = {
    'neopt': 'red',
    'opt_m': 'blue',
    'blas': 'green',
    'n2_opt_m': 'darkblue',
    'n2_blas': 'darkgreen'
}

# 4. Parcurgerea fiecărui test și trasarea liniei aferente
for test_type in df['Test'].unique():
    subset = df[df['Test'] == test_type]
    
    plt.plot(
        subset['N'], 
        subset['Time'], 
        marker='o', 
        linewidth=2,
        label=test_type, 
        color=colors.get(test_type, 'black')
    )

# 5. APLICAREA SCĂRII LOGARITMICE (Modificarea principală)
plt.yscale('log')

# 6. Adăugarea titlului și a etichetelor
plt.title('Performance Comparison (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Elements (N)', fontsize=12)
plt.ylabel('Time (log seconds)', fontsize=12)

# 7. Adăugarea grilei și a legendei
# Pentru scara logaritmică, which='both' ajută la afișarea liniilor intermediare (minor ticks)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(title='Test Version', fontsize=10)

plt.tight_layout()

# 8. Salvarea imaginii SVG și afișarea graficului
output_filename = 'plot_log_n2.svg'
plt.savefig(output_filename, format='svg', dpi=300, bbox_inches='tight')
print(f"Graficul a fost salvat cu succes ca '{output_filename}' în directorul curent.")

plt.show()
