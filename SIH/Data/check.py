import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Data\Soil_Fertility_with_Score.csv")

df = df.drop(columns=["index"], errors="ignore")

X_df = df.drop(columns=["Fertility"])
y = df["Fertility"].values

feature_names = X_df.columns.tolist()

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X_df)  # shape: (n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)                # compute RMSE manually
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Random Forest Regression Results")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("\nFeature importances (descending):")
for i in sorted_idx:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(8,6))
plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

sample = [[140, 10, 550, 7.5, 0.6, 0.7, 6.0, 0.25, 0.3, 0.8, 9.0, 0.12]]
sample_arr = np.array(sample).reshape(1, -1)
sample_imputed = imputer.transform(sample_arr)  # ensure same preprocessing
pred = rf_model.predict(sample_imputed)
print("\nPredicted Fertility Score for sample:", pred[0])


# joblib.dump(rf_model, "soil_fertility_rf.pkl")
# joblib.dump(imputer, "imputer.pkl")
# print("Model and imputer saved!")