import pandas as pd

df = pd.read_csv("fifa_ranking_2022-10-06.csv")

print(df.head())

print(df.info())

print("\nКоличество пропущенных значений в каждом столбце:")

missing_values = df.isnull().sum()
print(missing_values)

df['rank'].fillna(df['rank'].mean(), inplace=True)
df['previous_rank'].fillna(df['previous_rank'].mean(), inplace=True)
df['points'].fillna(df['points'].mean(), inplace=True)
df['previous_points'].fillna(df['previous_points'].mean(), inplace=True)

df['team'].fillna(df['team'].mode()[0], inplace=True)
df['team_code'].fillna(df['team_code'].mode()[0], inplace=True)
df['association'].fillna(df['association'].mode()[0], inplace=True)

print("\nПропущенные значения после заполнения:")
print(df.isnull().sum())