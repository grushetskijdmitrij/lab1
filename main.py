import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv("01_company_info.csv")

print("Первые строки исходного датасета:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

numeric_features = [
    'fiftyDayAverageChange',
    'fiftyDayAverageChangePercent',
    'twoHundredDayAverageChange',
    'twoHundredDayAverageChangePercent',
    'priceEpsCurrentYear',
    'trailingPegRatio'
]

categorical_features = [
    'country',
    'sector'
]

id_features = [
    'symbol'
]

selected_columns = numeric_features + categorical_features + id_features

df = df[selected_columns]

print("\nПосле выбора столбцов:")
print(df.head())

print("\nПропуски:")
print(df.isnull().sum())

for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    df[col] = df[col].fillna("Unknown")

print("\nПосле обработки пропусков:")
print(df.isnull().sum())


scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

le_country = LabelEncoder()
le_sector = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])
df['sector'] = le_sector.fit_transform(df['sector'])

print(dict(enumerate(le_country.classes_)))

print(dict(enumerate(le_sector.classes_)))

print(df.head())

df.to_csv("processed_companies.csv", index=False)