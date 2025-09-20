import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Odisha_Crop_Recommendation.csv")

le_season = LabelEncoder()
df['Season'] = le_season.fit_transform(df['Season'])

X = df.drop(columns=['Recommended Crop'])
y = df['Recommended Crop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

def recommend_crop(input_data, top_n=3):
    """
    input_data: dict with keys ['Season', 'Rainfall(mm)', 'Temperature(°C)', 'Area(ha)']
    Example: {"Season": "Rabi", "Rainfall(mm)": 500, "Temperature(°C)": 26, "Area(ha)": 5}
    """
    input_data["Season"] = le_season.transform([input_data["Season"]])[0]
    features = pd.DataFrame([input_data])
    
    probs = rf_model.predict_proba(features)[0]
    
    top_indices = np.argsort(probs)[-top_n:][::-1]
    top_crops = [(rf_model.classes_[i], probs[i]) for i in top_indices]
    return top_crops

example = {"Season": "Kharif", "Rainfall(mm)": 1200, "Temperature(°C)": 20, "Area(ha)": 5}
print(recommend_crop(example, top_n=3))
