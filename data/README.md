# 📊 BPJS Fraud Detection - Synthetic Dataset

## 📝 Overview

Dataset sintetis ini dibuat untuk penelitian dan pengembangan sistem deteksi fraud pada klaim BPJS Kesehatan. Dataset mensimulasikan transaksi klaim yang realistis dengan berbagai pola fraud yang umum terjadi dalam sistem healthcare claims.

**⚠️ DISCLAIMER**: Dataset ini adalah **SYNTHETIC DATA** yang di-generate secara artificial untuk tujuan penelitian. Data ini BUKAN data real BPJS Kesehatan dan tidak mengandung informasi pribadi atau sensitif.

---

## 📁 File Structure

```
dataset/
├── main.py          
├── generate_synthetic_bpjs.py      
├── metadata.json                      
└── README.md                          
```

---

## 📋 Dataset Specifications

| Property | Value |
|----------|-------|
| **Total Records** | 100,000 (default, configurable) |
| **Time Period** | 2024 (configurable) |
| **Fraud Ratio** | 3% (configurable) |
| **Format** | CSV, Parquet |
| **Encoding** | UTF-8 |

---

## 🗂️ Data Schema

### **Core Identifiers**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `claim_id` | string | Unique claim identifier | C00001234 |
| `episode_id` | string | Episode/visit identifier | E00001234 |
| `participant_id` | string | Participant/patient identifier | P00012345 |
| `nik_hash` | string | Hashed National ID (for identity fraud detection) | NIK0012345678 |
| `faskes_id` | string | Healthcare facility identifier | F00123 |
| `dpjp_id` | string | Doctor/provider identifier | D00456 |

### **Demographic & Geographic**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `age` | integer | Patient age | 45 |
| `sex` | string | Patient gender | M/F |
| `provinsi` | string | Province | DKI Jakarta |
| `kabupaten` | string | Regency/city | Jakarta Selatan |
| `faskes_level` | string | Facility level | FKTP / FKRTL |

### **Clinical Information**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `tgl_pelayanan` | date | Service date (ISO format) | 2024-03-15 |
| `kode_icd10` | string | ICD-10 diagnosis code | J00 (Common cold) |
| `kode_prosedur` | string | Procedure code | 89.03 |
| `jenis_pelayanan` | string | Service type | Rawat Jalan / Rawat Inap |
| `room_class` | string | Room class (inpatient only) | Kelas I / Kelas II / Kelas III / VIP |
| `lama_dirawat` | integer | Length of stay (days) | 5 |

### **Financial Data**

| Column | Type | Description | Example | Unit |
|--------|------|-------------|---------|------|
| `billed_amount` | integer | Total billed amount | 1500000 | IDR (Rupiah) |
| `paid_amount` | integer | Amount paid by BPJS | 1200000 | IDR |
| `selisih_klaim` | integer | Difference (billed - paid) | 300000 | IDR |
| `drug_cost` | integer | Medication cost | 500000 | IDR |
| `procedure_cost` | integer | Procedure cost | 800000 | IDR |

### **Behavioral Metrics**

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `visit_count_30d` | integer | Number of visits in last 30 days | 0-50 |
| `kapitasi_flag` | boolean | Capitation payment indicator | True/False |
| `referral_flag` | boolean | Has referral | True/False |
| `referral_to_same_facility` | boolean | Suspicious referral pattern | True/False |

### **Computed Features**

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `claim_ratio` | float | Ratio of billed to paid | billed_amount / paid_amount |
| `drug_ratio` | float | Drug cost as fraction of total | drug_cost / billed_amount |
| `procedure_ratio` | float | Procedure cost fraction | procedure_cost / billed_amount |
| `provider_claim_share` | float | Provider's share of monthly claims | provider_claims / total_monthly_claims |

### **Fraud Labels** (Ground Truth)

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `fraud_flag` | integer | Fraud indicator | 0 (Normal) / 1 (Fraud) |
| `fraud_type` | string | Type of fraud pattern | See [Fraud Types](#fraud-types) |
| `severity` | string | Fraud severity level | ringan / sedang / berat / none |
| `evidence_type` | string | Evidence source | system_anom / audit / whistleblower / none |

---

## 🔍 Fraud Types

Dataset includes 6 realistic fraud patterns:

### 1. **Phantom Claims** (25% of fraud)
- **Description**: Claims for services never rendered
- **Characteristics**:
  - High billed amounts (2-6x normal)
  - Zero or very low visit count (0-1 visits in 30 days)
  - No supporting documentation
- **Severity**: Berat (Critical)
- **Detection**: System anomaly, Audit

### 2. **Upcoding** (25% of fraud)
- **Description**: Billing for more expensive services than provided
- **Characteristics**:
  - Billed amount inflated 1.5-3x
  - Minor diagnosis mapped to expensive procedures
  - Clinical inconsistencies
- **Severity**: Sedang/Berat (Medium/High)
- **Detection**: Audit

### 3. **Unbundling** (15% of fraud)
- **Description**: Splitting one procedure into multiple claims
- **Characteristics**:
  - Multiple claims for same episode
  - Overlapping procedure codes
  - Total cost exceeds expected amount
- **Severity**: Sedang (Medium)
- **Detection**: System anomaly

### 4. **Prolonged LOS** (15% of fraud)
- **Description**: Unnecessary extended hospitalization
- **Characteristics**:
  - Length of stay 15-60 days (unusually long)
  - Inflated billing (2-4x normal)
  - Simple diagnosis with complex treatment
- **Severity**: Berat (Critical)
- **Detection**: Audit

### 5. **Identity Fraud** (10% of fraud)
- **Description**: Using stolen/shared identities
- **Characteristics**:
  - Same NIK hash across multiple participant IDs
  - Impossible age gaps for same identity
  - Concurrent claims at different locations
- **Severity**: Berat (Critical)
- **Detection**: Whistleblower

### 6. **Inflated Drugs** (10% of fraud)
- **Description**: Excessive medication charges
- **Characteristics**:
  - Drug cost 2-10x normal
  - Quantity mismatches
  - Drug ratio > 70% of total bill
- **Severity**: Sedang/Berat (Medium/High)
- **Detection**: System anomaly

---

## 📊 Data Distribution

### **Statistical Distributions**

Dataset uses realistic statistical distributions:

| Feature | Distribution | Parameters |
|---------|--------------|------------|
| `billed_amount` | Gamma | shape=2-3, scale=150k-500k (outpatient/inpatient) |
| `visit_count_30d` | Negative Binomial | mean=2.5, dispersion=3.0 |
| `lama_dirawat` | Poisson | lambda=5 (for inpatient) |
| `age` | Mixture | 30% young (0-20), 50% adult (20-60), 20% elderly (60+) |
| `paid_amount` | Uniform ratio | 0.7-1.0 of billed_amount |

### **Class Distribution**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal Claims | 97,000 | 97% |
| Fraud Claims | 3,000 | 3% |

**Fraud Type Distribution** (within fraud cases):

| Fraud Type | Count | Percentage |
|------------|-------|------------|
| Phantom | 750 | 25% |
| Upcoding | 750 | 25% |
| Unbundling | 450 | 15% |
| Prolonged LOS | 450 | 15% |
| Identity | 300 | 10% |
| Inflated Drugs | 300 | 10% |

---

## 🌍 Geographic Distribution

### **Provinces Included**

- DKI Jakarta
- Jawa Barat
- Jawa Tengah
- Jawa Timur
- Sumatera Utara
- Banten
- Sulawesi Selatan
- Kalimantan Timur
- Bali
- Riau

### **Facility Level Distribution**

| Level | Description | Percentage |
|-------|-------------|------------|
| FKTP | Primary care (Puskesmas, Klinik) | 70% |
| FKRTL | Secondary/Tertiary (Hospitals) | 30% |

---

## 🔧 Data Generation Process

### **Generation Pipeline**

```
1. Generate Providers (500 default)
   ↓
2. Generate Participants (50,000 default)
   ↓
3. Assemble Base Claims (100,000 default)
   ├─ Sample providers & participants
   ├─ Generate service types & amounts
   ├─ Assign dates & diagnoses
   └─ Calculate costs
   ↓
4. Inject Fraud Patterns (3% default)
   ├─ Select fraud indices
   ├─ Apply fraud transformations
   └─ Label fraud types
   ↓
5. Feature Engineering
   ├─ Compute ratios
   ├─ Calculate provider metrics
   └─ Add temporal features
   ↓
6. Export (CSV + Parquet + Metadata)
```

### **Reproducibility**

Dataset generation is fully reproducible using random seeds:

```python
# Generate identical dataset
python generate_synthetic_bpjs.py \
  --n_rows 1000 \
  --fraud_ratio 0.05 \
  --year 2024 \
  --seed 42
```

---

## 📈 Data Quality Checks

### **Validation Rules**

✅ All claims have `paid_amount <= billed_amount`  
✅ No negative monetary values  
✅ Drug + procedure costs ≤ billed amount  
✅ Inpatient claims have `lama_dirawat > 0`  
✅ Outpatient claims have `lama_dirawat = 0`  
✅ Fraud ratio within ±20% of target  
✅ All dates within specified year  
✅ No missing values in critical fields  

---

## 💡 Usage Examples

### **Load Data (Python)**

```python
import pandas as pd

# Load CSV
df = pd.read_csv('synthetic_bpjs_claims.csv')

# Load Parquet (faster, compressed)
df = pd.read_parquet('synthetic_bpjs_claims.parquet')

# Check shape
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Check fraud distribution
print(df['fraud_flag'].value_counts())
```

### **Basic Exploration**

```python
# Summary statistics
print(df[['billed_amount', 'paid_amount', 'lama_dirawat']].describe())

# Fraud by type
print(df[df['fraud_flag']==1]['fraud_type'].value_counts())

# Fraud by province
fraud_by_province = df.groupby('provinsi')['fraud_flag'].mean()
print(fraud_by_province.sort_values(ascending=False))
```

### **Split Train/Test**

```python
from sklearn.model_selection import train_test_split

# Time-based split (recommended)
df['date'] = pd.to_datetime(df['tgl_pelayanan'])
df = df.sort_values('date')
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Random split (with stratification)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['fraud_flag'],
    random_state=42
)
```

---

## 🔐 Data Privacy

### **Privacy-Preserving Features**

- ✅ **No Real Data**: Entirely synthetic, generated algorithmically
- ✅ **No PII**: No names, addresses, or real NIK numbers
- ✅ **Hashed Identifiers**: All IDs are random or hashed
- ✅ **Anonymized**: Cannot be linked to real individuals or institutions
- ✅ **Safe for Public Use**: Suitable for research, education, and open-source projects

### **GDPR/Privacy Compliance**

This synthetic dataset:
- Does NOT contain personal data as defined by GDPR
- Does NOT require consent or data protection impact assessment
- Can be freely shared for research purposes
- Is exempt from data protection regulations (synthetic data exclusion)

---

## 📚 Use Cases

### **1. Machine Learning Research**
- Train fraud detection models (XGBoost, Random Forest, Neural Networks)
- Develop feature engineering techniques
- Benchmark algorithm performance
- Study class imbalance handling

### **2. Graph Analytics**
- Detect fraud rings and collusion networks
- Analyze referral patterns
- Study provider-patient relationships
- Community detection algorithms

### **3. Education & Training**
- Healthcare fraud detection courses
- Data science bootcamps
- ML model deployment tutorials
- System integration examples

### **4. System Development**
- Test fraud detection pipelines
- Develop audit dashboards
- Build alerting systems
- API development and testing

---

## 🔄 Dataset Versions

| Version | Date | Records | Changes |
|---------|------|---------|---------|
| v1.0 | 2024-10 | 100,000 | Initial release |

---

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{bpjs_fraud_synthetic_2024,
  title={BPJS Healthcare Claims Fraud Detection - Synthetic Dataset},
  author={[Your Name/Organization]},
  year={2024},
  version={1.0},
  url={[Your Repository URL]}
}
```

---

## 🤝 Contributing

Found issues or have suggestions for improving the dataset?

1. Check existing issues
2. Create a new issue with details
3. Submit pull request with improvements

---

## 📄 License

This dataset is released under **[MIT License / CC BY 4.0]**.

You are free to:
- ✅ Use for commercial purposes
- ✅ Modify and distribute
- ✅ Use in research and publications
- ✅ Include in products and services

**Attribution**: Please provide appropriate credit when using this dataset.

---

## 📞 Contact & Support

- **Email**: alwanrahmana@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/alwanrahmana/
---


**Last Updated**: October 2024  
**Dataset Version**: 1.0  
**Format Version**: 1.0
