import pandas as pd

# Load the CSV file
df = pd.read_csv('zaraSales.csv')

# Save as Excel (index=False prevents adding an extra row-number column)
df.to_excel('zaraSales.xlsx', index=False)
