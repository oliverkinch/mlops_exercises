import pandas as pd
import matplotlib.pyplot as plt

results_path = "reports/time_results.txt"

df = pd.read_csv(results_path, index_col=False)

x = df.n.values
y = df.m.values
yerr = df.s.values

plt.errorbar(x, y, yerr)
plt.show()
