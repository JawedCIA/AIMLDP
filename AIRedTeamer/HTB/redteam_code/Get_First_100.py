import pandas as pd

# Read the original CSV file
df = pd.read_csv("train_2.csv")
r=50
# Get the first 100 rows
first_100 = df.tail(r)

# Save to a new CSV file
first_100.to_csv('poison_data_50_tail_2.csv', index=False)

print("Saved first {r} rows ")
