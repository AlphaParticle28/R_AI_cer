# 🏎️ F1 Tire Degradation Prediction & Pit Strategy System

An intelligent tire wear prediction framework for Formula 1 racing, leveraging **deep neural networks** and **real-time telemetry data fusion** to integrate thermal dynamics, material properties, and mechanical loading parameters.

Supports **end-to-end workflows** from K-value prediction to real-time tire wear monitoring and pit stop strategy optimization.

---

## 🌟 Project Overview
This system revolutionizes F1 race strategy by combining **machine learning** with **physics-based tire modeling** to predict optimal pit stop timing and prevent tire failures.

### **Key Capabilities**
- **K-Value Prediction** – Neural network model predicting tire wear coefficient with **94.7% R² accuracy**
- **Real-Time Monitoring** – Continuous tire degradation tracking using live telemetry data
- **Pit Strategy Optimization** – Automated recommendations for optimal pit stop timing
- **Multi-Tire Analysis** – Individual monitoring of all 4 tire positions with compound-specific modeling
- **Safety Alerts** – Dynamic threshold monitoring with **8-minute advance warnings**

---

## 🗂️ Directory Structure
```plaintext
Racecar/
│
├── Dataset/
│   ├── RaceCarUsefulDataDriver1.xlsx    # Primary telemetry dataset (12,100+ samples)
│   ├── toFindK.xlsx                     # K-value training/validation data
│   └── TyreDegradation.csv              # Original dataset source
│
├── k-estimator.ipynb                    # K-value prediction model training
├── predictor.py                         # Production ML inference system
├── time_estimator.ipynb                 # Tire wear monitoring & pit strategy
└── README.md                            # Project documentation
```

---

## 💾 Workflow Description

### **1. K-Value Model Development**
**File:** `k-estimator.ipynb`
- Trains **17-feature neural network** using tire temperature, force, compound data
- Incorporates **thermodynamic modeling** and **Archard's wear law**
- Achieves **94.7% R² accuracy**
- Saves trained model (`.keras`), scalers (`.pkl`), and config (`.json`)

---

### **2. Production Inference System**
**File:** `predictor.py`
- Loads trained model with **automated feature engineering**
- Handles **one-hot encoded tire compounds (C1–C5)**
- Processes **real-time telemetry data**
- Provides **standardized API** for K-value predictions

---

### **3. Real-Time Tire Monitoring**
**File:** `time_estimator.ipynb`
- Integrates K-value predictions with **live race telemetry**
- Calculates **dynamic tire wear** using physics-based degradation model
- Generates **pit stop recommendations** with safety threshold monitoring
- Tracks individual tire degradation across **all 4 positions**

---

## 📊 Data Management

### **Primary Datasets**
- **`RaceCarUsefulDataDriver1.xlsx`** – Full telemetry dataset
- **`toFindK.xlsx`** – K-value training/validation data
- **`TyreDegradation.csv`** – Reference degradation patterns

### **Model Artifacts**
- `k_predictor_model.keras` – Trained neural network (6 layers, 79,553 parameters)
- `feature_scaler.pkl` – Input scaler
- `target_scaler.pkl` – Output scaler
- `model_config.json` – Model metadata

---

## 🧮 Technical Architecture

### **Neural Network Specifications**
| Property        | Details |
|-----------------|---------|
| **Architecture** | 64 → 128 → 256 → 128 → 32 → 1 neurons |
| **Input Features** | 17 (12 numerical + 5 one-hot encoded) |
| **Activations** | ReLU, Linear, Tanh |
| **Optimizer** | Adam (lr = 0.001) |
| **Performance** | R² = 0.947, MAE = 0.076, RMSE = 0.23 |

---

## 📦 Prerequisites

### **System Requirements**
- Python **3.10+**
- Jupyter Notebook
- TensorFlow/Keras 2.x
- scikit-learn, pandas, numpy, matplotlib

### **Required Libraries**
```bash
pip install tensorflow tf-keras scikit-learn pandas numpy matplotlib joblib openpyxl
```

---

## 🛠️ Installation Instructions

### **1. Environment Setup**
```bash
# Clone repository
git clone https://github.com/AlphaParticle28/R_AI_cer
cd Racecar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install tensorflow tf-keras scikit-learn pandas numpy matplotlib joblib openpyxl
```

### **2. Data Preparation**
- Place all Excel files in `Dataset/`
- Verify column names and integrity
- Handle missing values

---

## 🚦 How to Run

### **Step 1 – Train K-Value Prediction Model**
```bash
jupyter notebook k-estimator.ipynb
```

### **Step 2 – Test Production Inference**
```python
from predictor import KValuePredictor

predictor = KValuePredictor()
sample_data = {
    "tire1": 100, "tire2": 120, "tire3": 80, "tire4": 90,
    "humidity1": 55, "temp_surr1": 15, "surface_rougness1": 1.5,
    "force1": 30000, "fric_coeff1": 1.0, "v1": 160,
    "t1": 0.1, "tire_type_encoded1": "C3", "E": 4000000
}
k_value = predictor.predict(sample_data)
print(f"Predicted K value: {k_value:.2e}")
```

### **Step 3 – Real-Time Tire Monitoring**
```bash
jupyter notebook time_estimator.ipynb
```

---

## ⚡ Usage Scenarios
- **Race Strategy Optimization** – Monitor live tire wear and get pit recommendations
- **Model Retraining** – Add new data and retrain for better accuracy
- **Performance Analysis** – Compare degradation across compounds and conditions

---

## 🏁 Key Features
| Feature | Description | Benefit |
|---------|-------------|---------|
| **K-Value Prediction** | ML-based tire wear coefficient estimation | 94.7% accuracy |
| **Multi-Compound Support** | Supports C1–C5 compounds | Full coverage |
| **Real-Time Processing** | Batch process 12k+ samples | Live monitoring |
| **Physics Integration** | ML + Archard’s wear law | Scientifically sound |
| **Safety Monitoring** | Threshold alerts | Prevents failures |
| **Individual Tire Tracking** | Per-tire analysis | Precise strategy |

---

## 📈 Model Performance
- **Training Accuracy**: R² = 0.947
- **Validation Loss**: 0.053 (low overfitting)
- **Prediction Speed**: ~200+ predictions/sec
- **Feature Growth**: 13 → 17 via one-hot encoding
- **Real-World Use**: Processes live F1 telemetry

---

## 🔬 Technical Implementation

**Tire Wear Calculation**
```python
wear_increment = (K_predicted * Force * Velocity * time_delta * safety_factor) / tire_hardness
```

**Temperature-Dependent Hardness**
```python
adjusted_hardness = base_hardness + slope * (tire_temp - 23)
elastic_modulus = f(adjusted_hardness)  # Physics-based conversion
```

**Pit Stop Decision Logic**
```python
time_remaining = ((weight_limit / current_wear) - 1) * elapsed_time
if time_remaining <= 8:
    trigger_pit_stop_alert()
```
