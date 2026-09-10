import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# ------------------ Load Dataset ------------------
df = pd.read_csv("fraud_detection_dataset_with_pca.csv")

# ------------------ Features & Target ------------------
features = ["Amount", "IP_Location", "Phone_Number", "V1", "V2", "V3", "V4", "V5"]
X = df[features].copy()
y = df["Is_Fraud"]

# ------------------ Encode Categorical Features ------------------
le_location = LabelEncoder()
le_phone = LabelEncoder()

X["IP_Location"] = le_location.fit_transform(X["IP_Location"])
X["Phone_Number"] = le_phone.fit_transform(X["Phone_Number"])

# Save encoders
joblib.dump(le_location, "encoder_location.pkl")
joblib.dump(le_phone, "encoder_phone.pkl")

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Apply SMOTE ------------------
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ------------------ Feature Scaling ------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# ------------------ Save Mean PCA Values ------------------
pca_mean_values = X.mean().tolist()[3:8]
joblib.dump(pca_mean_values, "pca_mean_values.pkl")
print("✅ PCA mean values saved successfully.")

# ------------------ Train Model ------------------
model = RandomForestClassifier(
    n_estimators=100,  # Reduced from 200 to prevent overfitting
    random_state=42,
    class_weight="balanced",
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train_scaled, y_train_res)

# Save the trained model
joblib.dump(model, "fraud_model.pkl")

print("✅ Model, scaler, and encoders saved successfully.")
