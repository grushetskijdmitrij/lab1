import pandas as pd

df = pd.read_csv("processed_companies.csv")

print(df.head())

median_value = df['fiftyDayAverageChangePercent'].median()

df['target_class'] = (df['fiftyDayAverageChangePercent'] > median_value).astype(int)

print("\nРаспределение классов:")
print(df['target_class'].value_counts())