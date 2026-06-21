import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All libraries loaded ✓")

# Load both sheets from the Excel file
df_09 = pd.read_excel('/Users/gargik/Documents/Customer Churn Project/Data/online_retail_II.xlsx', sheet_name='Year 2009-2010')
df_10 = pd.read_excel('/Users/gargik/Documents/Customer Churn Project/Data/online_retail_II.xlsx', sheet_name='Year 2010-2011')

# Combine into one dataframe
df = pd.concat([df_09, df_10], ignore_index=True)

print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"\nColumn names:\n{df.columns.tolist()}")

