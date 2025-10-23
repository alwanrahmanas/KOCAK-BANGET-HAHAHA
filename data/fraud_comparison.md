# 🏥 BPJS Fraud Types - Complete Reference

## 📊 Fraud Type Comparison Table

| # | Fraud Type | Indonesian Name | % | Severity | Detection | Multiplier | Indicators |
|---|------------|-----------------|---|----------|-----------|------------|-----------|
| 1 | **Upcoding Diagnosis** | Upcoding Diagnosis | 15% | 🟡 Sedang-Berat | Audit | 1.5-2.5x | Changed ICD-10, clinical mismatch |
| 2 | **Phantom Billing** | Klaim Fiktif | 12% | 🔴 Berat | System/Audit | 2.5-6.0x | Visit count = 0-1, no documentation |
| 3 | **Cloning Claim** | Klaim Duplikat | 8% | 🔴 Berat | Audit | - | Duplicate patterns, same ICD/procedure |
| 4 | **Inflated Bill** | Tagihan Digelembungkan | 12% | 🟡 Sedang | System | 2.5-8.0x | High drug/alkes cost, qty mismatch |
| 5 | **Service Unbundling** | Pemecahan Paket | 10% | 🟡 Sedang | System | 2.0-4.0x | Multiple claims, same episode |
| 6 | **Self-Referral** | Rujukan Internal | 10% | 🟡 Sedang | Audit | 1.3-2.0x | Referral to same facility |
| 7 | **Repeat Billing** | Tagihan Berulang | 8% | 🔴 Berat | System | - | Duplicate submission, same service |
| 8 | **Prolonged LOS** | Perpanjangan Rawat | 10% | 🟡 Sedang | Audit | 2.0-4.0x | LOS 15-60 days, no medical reason |
| 9 | **Room Manipulation** | Manipulasi Kelas | 8% | 🟡 Sedang | Audit | 1.4-2.2x | Claimed higher room class |
| 10 | **Unnecessary Services** | Layanan Berlebihan | 5% | 🟢 Ringan-Sedang | Audit | 1.5-2.5x | High visit count (5-15/month) |
| 11 | **Fake License** | Izin Palsu | 2% | 🔴 Berat | Whistleblower | 1.2-2.0x | Invalid/expired license |

---

## 🎯 Detection Strategy Matrix

| Fraud Type | Primary Signal | Secondary Signal | Data Pattern | Alert Priority |
|------------|---------------|------------------|--------------|----------------|
| **Upcoding Diagnosis** | ICD-10 mismatch | High bill for minor condition | Diagnosis severity vs cost | 🟡 Medium |
| **Phantom Billing** | Zero visit count | Very high billing | No clinical documentation | 🔴 High |
| **Cloning Claim** | Identical patterns | Same ICD + procedure | Pattern matching across claims | 🔴 High |
| **Inflated Bill** | Drug cost ratio >70% | Quantity anomaly | Drug/alkes cost spike | 🟡 Medium |
| **Service Unbundling** | Multiple claims | Same episode ID | Overlapping procedures | 🟡 Medium |
| **Self-Referral** | Referral to same facility | No medical justification | Provider conflict of interest | 🟡 Medium |
| **Repeat Billing** | Duplicate claim ID | Same date + service | Exact duplicate in system | 🔴 High |
| **Prolonged LOS** | LOS >20 days | Minor diagnosis | LOS vs diagnosis severity | 🟡 Medium |
| **Room Manipulation** | Room class mismatch | Kelas I/VIP claim | Room class vs actual | 🟡 Medium |
| **Unnecessary Services** | Visit frequency >10/month | No clinical justification | Overutilization pattern | 🟢 Low |
| **Fake License** | Invalid SIP/license | Provider verification fail | License database mismatch | 🔴 High |

---

## 💰 Financial Impact Analysis

### **High Impact Fraud** (>2.5x multiplier)
```
Phantom Billing:     2.5-6.0x  → Avg loss: Rp 3-8 juta per claim
Inflated Bill:       2.5-8.0x  → Avg loss: Rp 2-6 juta per claim
Service Unbundling:  2.0-4.0x  → Avg loss: Rp 1.5-4 juta per claim
Prolonged LOS:       2.0-4.0x  → Avg loss: Rp 2-5 juta per claim
```

### **Medium Impact Fraud** (1.5-2.5x multiplier)
```
Upcoding Diagnosis:  1.5-2.5x  → Avg loss: Rp 500rb-2 juta per claim
Room Manipulation:   1.4-2.2x  → Avg loss: Rp 400rb-1.5 juta per claim
Unnecessary Svc:     1.5-2.5x  → Avg loss: Rp 300rb-1 juta per claim
```

### **Low-Medium Impact Fraud** (1.2-2.0x multiplier)
```
Self-Referral:       1.3-2.0x  → Avg loss: Rp 300rb-1 juta per claim
Fake License:        1.2-2.0x  → Avg loss: Rp 200rb-800rb per claim
```

---

## 🔍 Feature Engineering for ML Detection

### **Features by Fraud Type**

| Fraud Type | Key Features | Feature Engineering |
|------------|-------------|---------------------|
| **Upcoding Diagnosis** | `claim_ratio`, `icd10_severity_score` | ICD-10 severity mapping, diagnosis-cost correlation |
| **Phantom Billing** | `visit_count_30d`, `claim_ratio`, `provider_claim_share` | Visit history, documentation flags |
| **Cloning Claim** | `pattern_similarity`, `diagnosis_frequency` | Cosine similarity, pattern matching |
| **Inflated Bill** | `drug_ratio`, `drug_cost`, `qty_per_unit_cost` | Drug cost benchmarking, quantity analysis |
| **Service Unbundling** | `procedure_overlap_score`, `episode_claim_count` | Procedure bundling rules, episode grouping |
| **Self-Referral** | `referral_to_same_facility`, `provider_facility_count` | Provider-facility network graph |
| **Repeat Billing** | `claim_hash_duplicate`, `temporal_proximity` | Hash matching, time window analysis |
| **Prolonged LOS** | `lama_dirawat`, `los_diagnosis_ratio`, `ventilator_days` | LOS benchmarking by diagnosis |
| **Room Manipulation** | `room_class_encoded`, `room_cost_ratio` | Room class verification, cost benchmarking |
| **Unnecessary Services** | `visit_count_30d`, `procedure_count`, `utilization_rate` | Visit frequency analysis, procedure necessity |
| **Fake License** | `provider_license_status`, `license_expiry` | License database integration |

---

## 📈 ML Model Training Recommendations

### **Stratification Strategy**
```python
# Stratify by severity for balanced training
severity_weights = {
    'berat': 3.0,      # Critical cases - oversample
    'sedang': 1.5,     # Medium cases - normal
    'ringan': 1.0      # Low cases - undersample
}

# Or stratify by fraud type
fraud_type_weights = {
    'phantom_billing': 2.5,
    'fake_license': 3.0,
    'repeat_billing': 2.5,
    'upcoding_diagnosis': 1.5,
    # ... etc
}
```

### **Feature Importance Priority**
```
Top features for multi-fraud detection:
1. claim_ratio (universal)
2. visit_count_30d (phantom, cloning)
3. drug_ratio (inflated bill)
4. lama_dirawat (prolonged LOS)
5. referral_to_same_facility (self-referral)
6. provider_claim_share (all types)
7. procedure_overlap_score (unbundling)
```

### **Model Architecture Suggestions**
```
Ensemble approach:
1. XGBoost classifier (main model)
2. Isolation Forest (anomaly detection)
3. Graph Neural Network (network-based fraud)
4. Rule-based filters (pre-screening)

Multi-task learning:
- Task 1: Binary fraud detection (fraud_flag)
- Task 2: Fraud type classification (11 classes)
- Task 3: Severity prediction (ringan/sedang/berat)
```

---

## 🎓 Educational Use Cases

### **For Students/Researchers**
1. **Binary Classification**: Train model to detect fraud vs normal
2. **Multi-class Classification**: Predict fraud type (11 classes)
3. **Imbalanced Learning**: Handle 3% fraud ratio
4. **Cost-sensitive Learning**: Weight by financial impact
5. **Explainable AI**: Use SHAP to explain predictions

### **For Practitioners**
1. **Audit Prioritization**: Rank claims by fraud probability
2. **Provider Risk Scoring**: Identify high-risk providers
3. **Network Analysis**: Detect fraud rings and collusion
4. **Real-time Scoring**: Deploy as API for live claims
5. **Dashboard Development**: Build monitoring system

---

## 📚 Real-World Mapping

### **How These Fraud Types Map to Actual BPJS Cases**

| Fraud Type | Real-World Prevalence | Common Scenarios |
|------------|----------------------|------------------|
| **Upcoding** | ⭐⭐⭐⭐⭐ Very Common | Private hospitals, chronic disease claims |
| **Phantom** | ⭐⭐⭐⭐ Common | Small clinics, rural areas |
| **Inflated Bill** | ⭐⭐⭐⭐⭐ Very Common | Drug procurement
