import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix

df = pd.read_csv("processed_companies.csv")

print("Размер датасета:", df.shape)

target = 'fiftyDayAverageChangePercent'

X = df.drop(columns=[target, 'symbol'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2)

X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Реальные:", y_test.values[:10])
print("Предсказанные:", y_pred[:10])

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

median_value = df['fiftyDayAverageChangePercent'].median()

df['target_class'] = (df['fiftyDayAverageChangePercent'] > median_value).astype(int)

X = df.drop(columns=['target_class', 'symbol'])
y = df['target_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Реальные:", y_test.values[:10])
print("Предсказанные:", y_pred[:10])

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))