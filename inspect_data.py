import pandas as pd

# Inspect Music Info.csv
df1 = pd.read_csv('Music Info.csv', nrows=5)
print("=== Music Info.csv ===")
print("Columns:", df1.columns.tolist())

# Check for valence and arousal
print("\nFirst row features:")
print(df1.iloc[0])

# Inspect User Listening History
df2 = pd.read_csv('User Listening History.csv', nrows=5)
print("\n=== User Listening History.csv ===")
print("Columns:", df2.columns.tolist())
print(df2.iloc[0])
