# 🤖 IoT Predictive Maintenance for Automated Sorting Robots

**Predicting drive motor catastrophic failures 24 hours in advance using 10,000+ hours of IoT sensor telemetry and time-series machine learning.**

## 📌 Project Overview
---

At **AeroLogistics**, global fulfillment centers rely on high-speed automated sorting robots. Recently, primary drive motors in these robots have been unexpectedly overheating and failing, causing catastrophic warehouse shutdowns that cost **$50,000 per minute** in delayed shipments.

This project transitions maintenance operations from a **reactive strategy** (fixing motors after they break) to a **proactive strategy** (predicting failures 24 hours in advance)[cite: 1]. Using continuous IoT sensor telemetry, we engineer multi-hour rolling time-series features, shift target labels to eliminate lead-time bias, and train a Random Forest classifier optimized for severe class imbalance.

---

## 🛠️ Key Technical Challenges Overcome

1. **The Windowing Trap (Feature Engineering):** Single instantaneous spikes do not indicate breakdown; sustained degradation does. We computed **12-hour rolling mathematical means** for temperature and **12-hour rolling standard deviations** for vibration to capture sustained degradation over time.
2. **The Lead-Time Trap (Target Shifting):** Predicting failure at the exact moment it occurs gives mechanics zero time to intervene. We applied `.shift(-24)` to move target labels backward, forcing the AI to learn pre-failure patterns **24 hours in advance**.
3. **The Data Leakage Trap:** Standard random train/test shuffling lets time-series models "look into the future" to guess the past. We implemented a strict **80/20 chronological partition** without shuffling.
4. **Explainable AI (XAI):** Extracted mathematical feature importances to identify whether raw vibration, rolling thermal buildup, or voltage drops serve as the primary leading indicators of failure.

---

## 📊 Performance & Key Results

* **Failure Recall (24h Notice):** **66.7%** — Successfully caught **2 out of 3 actual failure events** in unseen chronological test data 24 hours before they occurred[cite: 1, 6].
* **Overall Accuracy:** **99.2%** on chronological test data.
* **Controlled Sensitivity:** Balanced class weighting generates minor false alarms.In warehouse operations, spending 10 minutes on a precautionary inspection is far cheaper than suffering an unpredicted **$50,000/minute** breakdown.

### Feature Importance (XAI)
Based on Random Forest feature extraction:
1. **`Sensor_Vibration_mm` (38.7%):** Primary leading indicator of loose components and bearing wear 24 hours prior.
2. **`Sensor_Temp_C` (27.2%) & `Temp_Rolling_Mean_12h` (26.2%):** Combined thermal buildup accounts for **53.4%** of predictive power, proving sustained friction drives failure.
3. **`Vib_Rolling_Std_12h` (4.8%) & `Sensor_Voltage_V` (3.1%):** Minor supporting indicators.

---

## 📂 Dataset Architecture

The dataset (`iot_sensor_telemetry.csv`) contains **10,000 hours** of continuous synchronous sensor readings[cite: 1]:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Timestamp` | Datetime | Hourly continuous time-stamp |
| `Sensor_Temp_C` | Float | Internal drive motor temperature (°C) |
| `Sensor_Vibration_mm` | Float | Motor housing vibration amplitude (mm)|
| `Sensor_Voltage_V` | Float | Electrical input voltage (V)|
| `Catastrophic_Failure` | Binary | Boolean failure indicator ($0.15\%$ positive rate)|

---

## 🚀 Quick Start & Usage

## pip install pandas numpy scikit-learn matplotlib seaborn

💡 Maintenance RecommendationAutomated Vibration Alerts: Set automated sensory triggers on raw vibration (Sensor_Vibration_mm) spikes.   Thermal Trend Monitoring: Monitor the 12-hour rolling average temperature (Temp_Rolling_Mean_12h).   Operational Protocol: When an alert sounds, dispatch technicians during off-peak shifts to inspect motor bearings, utilizing the 24-hour lead time window to guarantee zero unplanned downtime[cite: 1].
👤 AuthorRehan Alam — Data Science & Applications Student, IIT Madras 
  Certifications: Data Science Internship & Training Certificates, Internship Studio (Oct 2026) 
