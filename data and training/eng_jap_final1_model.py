import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import unicodedata
import joblib

# === Load Dataset ===
df = pd.read_json("headings_output.json")

# === Add language script feature ===
def detect_script(text):
    for ch in text:
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return "japanese"
    return "latin"

df["script_type"] = df["text"].map(detect_script)

# === Drop unneeded columns ===
df.drop(columns=[col for col in ["filename"] if col in df.columns], inplace=True)

# === Convert booleans to integers ===
df["is_bold"] = df["is_bold"].astype(int)
df["is_caps"] = df["is_caps"].fillna(False).astype(int)  # Japanese will be 0

# === Extract RGB values ===
df[["r", "g", "b"]] = df["color_rgb"].str.extract(r"#(..)(..)(..)")
df["r"] = df["r"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
df["g"] = df["g"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
df["b"] = df["b"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
df.drop(columns=["color_rgb"], inplace=True)

# === One-hot encode font names (if available) ===
if "font_name" in df.columns:
    df = pd.get_dummies(df, columns=["font_name"], prefix="font")

# === One-hot encode script type (latin, japanese) ===
df = pd.get_dummies(df, columns=["script_type"], prefix="lang")

# === Clean column names for LightGBM ===
df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)

# === Deduplicate Columns (for repeated font names) ===
def deduplicate_columns(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            new_columns.append(f"{col}_{seen[col]}")
            seen[col] += 1
    return new_columns

df.columns = deduplicate_columns(df.columns)

# === Split Features and Labels ===
X = df.drop(columns=["label", "text"])  # Drop 'text' too if still there
y = df["label"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Train Model ===
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(model, "heading_classifier_lgbm2.pkl")
print("💾 Model saved to 'heading_classifier_lgbm1.pkl'")
