#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Bootstrapping ML Inference Service (LightGBM + FastAPI)"

# Update system packages
dnf update -y

# Install Python runtime
dnf install -y python3 python3-pip

# Upgrade pip and install required Python packages
pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy fastapi uvicorn pydantic

# Prepare working directory
APP_DIR="/opt/inference"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Write model training script
cat << 'PYEOF' > train.py
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("Creating synthetic classification dataset...")
features, labels = make_classification(
    n_samples=10000,
    n_features=20,
    random_state=7
)

X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=7
)

train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.08,
    'feature_fraction': 0.85,
    'verbose': -1,
}

print("Training LightGBM classifier...")
booster = lgb.train(
    lgb_params,
    train_set,
    num_boost_round=150,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(stopping_rounds=15)],
)

booster.save_model('booster.txt')
print("Booster saved to booster.txt")
PYEOF

# Write FastAPI inference server
cat << 'PYEOF' > server.py
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LightGBM Inference API", version="1.0")

booster = lgb.Booster(model_file='booster.txt')

class InputData(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def run_inference(payload: InputData):
    if len(payload.features) != 20:
        raise HTTPException(status_code=422, detail="Requires exactly 20 feature values")

    arr = np.array([payload.features])
    score = booster.predict(arr)[0]
    label = int(score >= 0.5)

    return {
        "score": float(score),
        "label": label,
    }
PYEOF

# Train the model
python3 train.py

# Register FastAPI as a systemd service
cat << 'SVCEOF' > /etc/systemd/system/inference-api.service
[Unit]
Description=LightGBM Inference API
After=network-online.target
Wants=network-online.target

[Service]
User=root
WorkingDirectory=/opt/inference
ExecStart=/usr/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable inference-api
systemctl start inference-api

echo "Setup complete — inference API is running on port 8000"
