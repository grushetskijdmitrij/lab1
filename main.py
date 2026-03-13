import pandas as pd

df = pd.read_csv("fifa_ranking_2022-10-06.csv")

print(df.head())

print(df.info())

print("\nКоличество пропущенных значений в каждом столбце:")

missing_values = df.isnull().sum()
print(missing_values)