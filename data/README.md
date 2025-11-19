# 📊 BPJS Fraud Detection - Synthetic Dataset

## 📝 Overview

Dataset sintetis ini dibuat untuk penelitian dan pengembangan sistem deteksi fraud pada klaim BPJS Kesehatan. Dataset mensimulasikan transaksi klaim yang realistis dengan berbagai pola fraud yang umum terjadi dalam sistem healthcare claims.

**⚠️ DISCLAIMER**: Dataset ini adalah **SYNTHETIC DATA** yang di-generate secara artificial untuk tujuan penelitian. Data ini BUKAN data real BPJS Kesehatan dan tidak mengandung informasi pribadi atau sensitif.

---

## 📁 File Structure

```
data/
├── fraud_comparison.md          
├── fraud_type.md
├── generate_synthetic.py
├── main.py     
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
| **File Size** | ~15-20 MB (CSV), ~5-8 MB (Parquet) |
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
| `fraud_type` | string | Type of fraud pattern | See [Fraud Types](https://github.com/alwanrahmanas/AI-fraud-detection/blob/main/data/fraud_type.md) |
| `severity` | string | Fraud severity level | ringan / sedang / berat / none |
| `evidence_type` | string | Evidence source | system_anom / audit / whistleblower / none |

---

## 🔍 Fraud Types

Dataset includes **11 realistic fraud patterns** commonly found in healthcare claims:

### 1. **Upcoding Diagnosis** (15% of fraud) 🔴
- **Description**: Mengubah kode diagnosis agar tarif klaim lebih tinggi dari seharusnya
- **Characteristics**:
  - Billed amount inflated 1.5-2.5x
  - ICD-10 code changed to more severe condition
  - Clinical inconsistencies between diagnosis and treatment
- **Severity**: Sedang-Berat (Medium-High)
- **Detection**: Audit
- **Example**: Pasien apendisitis tanpa komplikasi diklaim sebagai apendisitis perforasi

### 2. **Phantom Billing** (12% of fraud) 🔴
- **Description**: Mengajukan klaim atas layanan yang tidak pernah dilakukan (klaim fiktif)
- **Characteristics**:
  - High billed amounts (2.5-6x normal)
  - Zero or very low visit count (0-1 visits in 30 days)
  - No supporting documentation
- **Severity**: Berat (Critical)
- **Detection**: System anomaly, Audit
- **Example**: Klaim tindakan operasi yang tidak pernah dilakukan

### 3. **Cloning Claim** (8% of fraud) 🔴
- **Description**: Menyalin rekam medis atau data pasien lain untuk klaim baru
- **Characteristics**:
  - Duplicate patterns across different claims
  - Same diagnosis and procedure codes
  - Pattern matching anomalies
- **Severity**: Berat (Critical)
- **Detection**: Audit
- **Example**: Copy-paste data pasien lain ke klaim berbeda

### 4. **Inflated Bill** (12% of fraud) 🟡
- **Description**: Menggelembungkan tagihan obat atau alat kesehatan
- **Characteristics**:
  - Drug cost inflated 2.5-8x
  - Procedure cost inflated 1.5-3x
  - Quantity mismatches
- **Severity**: Sedang (Medium)
- **Detection**: System anomaly
- **Example**: Tagihan 4 vial obat padahal hanya diberikan 2

### 5. **Service Unbundling** (10% of fraud) 🟡
- **Description**: Memecah paket pelayanan menjadi beberapa klaim terpisah untuk menaikkan total pembayaran
- **Characteristics**:
  - Procedure cost inflated 2-4x
  - Multiple claims for same episode
  - Overlapping procedure codes
- **Severity**: Sedang (Medium)
- **Detection**: System anomaly
- **Example**: Pemeriksaan laboratorium yang seharusnya satu paket dibagi menjadi 4 klaim

### 6. **Self-Referral** (10% of fraud) 🟡
- **Description**: Merujuk pasien ke fasilitas milik sendiri tanpa alasan medis
- **Characteristics**:
  - Billed amount inflated 1.3-2x
  - Referral flag = True
  - Referral to same facility = True
  - Conflict of interest
- **Severity**: Sedang (Medium)
- **Detection**: Audit
- **Example**: Dokter merujuk pasien ke RS tempat ia juga bekerja

### 7. **Repeat Billing** (8% of fraud) 🔴
- **Description**: Menagih kembali klaim yang sudah dibayar
- **Characteristics**:
  - Duplicate claim submission
  - Same service billed multiple times
  - System-detectable duplicates
- **Severity**: Berat (Critical)
- **Detection**: System anomaly
- **Example**: Klaim rawat inap diajukan dua kali

### 8. **Prolonged LOS** (10% of fraud) 🟡
- **Description**: Memperpanjang lama rawat inap tanpa indikasi medis
- **Characteristics**:
  - Length of stay 15-60 days (unusually long)
  - Billed amount inflated 2-4x
  - Simple diagnosis with prolonged treatment
- **Severity**: Sedang (Medium)
- **Detection**: Audit
- **Example**: Pasien ventilator dipertahankan tanpa indikasi

### 9. **Room Manipulation** (8% of fraud) 🟡
- **Description**: Memanipulasi kelas perawatan untuk menaikkan biaya klaim
- **Characteristics**:
  - Billed amount inflated 1.4-2.2x
  - Claimed higher room class (Kelas I, VIP)
  - Actual class lower than claimed
- **Severity**: Sedang (Medium)
- **Detection**: Audit
- **Example**: Klaim kelas I padahal pasien dirawat di kelas II

### 10. **Unnecessary Services** (5% of fraud) 🟢
- **Description**: Memberikan layanan medis yang tidak sesuai indikasi
- **Characteristics**:
  - Procedure cost inflated 1.5-2.5x
  - High visit frequency (5-15 visits in 30 days)
  - Overutilization of services
- **Severity**: Ringan-Sedang (Low-Medium)
- **Detection**: Audit
- **Example**: Pemeriksaan berlebihan agar tagihan naik

### 11. **Fake License** (2% of fraud) 🔴
- **Description**: Menggunakan izin praktik atau izin operasional palsu
- **Characteristics**:
  - Billed amount inflated 1.2-2x
  - Invalid or expired license
  - Illegal medical practice
- **Severity**: Berat (Critical)
- **Detection**: Whistleblower
- **Example**: Dokter tanpa SIP aktif tetap mencatat layanan

---

## 📊 Fraud Distribution Summary

### **By Severity**

| Severity | Fraud Types | Total % |
|----------|-------------|---------|
| 🔴 **Berat (Critical)** | Phantom Billing, Cloning Claim, Repeat Billing, Fake License | 34% |
| 🟡 **Sedang (Medium)** | Upcoding, Inflated Bill, Unbundling, Self-Referral, Prolonged LOS, Room Manipulation | 61% |
| 🟢 **Ringan (Low)** | Unnecessary Services | 5% |

### **By Detection Method**

| Method | Fraud Types | % |
|--------|-------------|---|
| **System Anomaly** | Phantom Billing, Inflated Bill, Service Unbundling, Repeat Billing | 44% |
| **Audit** | Upcoding, Phantom, Cloning, Self-Referral, Prolonged LOS, Room Manipulation, Unnecessary | 54% |
| **Whistleblower** | Fake License | 2% |

### **By Financial Impact**

| Impact Level | Multiplier | Fraud Types |
|--------------|-----------|-------------|
| **Very High** | 2.5-8x | Phantom Billing, Inflated Bill |
| **High** | 2.0-4x | Service Unbundling, Prolonged LOS |
| **Medium** | 1.5-2.5x | Upcoding, Room Manipulation, Unnecessary Services |
| **Low** | 1.2-2x | Self-Referral, Fake License |

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
  --n_rows 100000 \
  --fraud_ratio 0.03 \
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
  author={Alwan Rahmana Subian},
  year={2025},
  version={1.0},
  url={https://github.com/alwanrahmanas/AI-fraud-detection}
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

- **LinkedIn**: https://www.linkedin.com/in/alwanrahmana/
- **Email**: alwanrahmana@gmail.com

---

## 🔗 Related Resources

- [Generation Script Documentation](../README.md)
- [Fraud Detection Model](../models/README.md)
- [API Documentation](../api/README.md)
- [Dashboard User Guide](../dashboard/README.md)

---

**Last Updated**: October 2024  
**Dataset Version**: 1.0  
**Format Version**: 1.0
