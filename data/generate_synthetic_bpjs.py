"""
BPJS Synthetic Claims Dataset Generator
========================================
Generate realistic BPJS-style healthcare claims with fraud patterns for ML experiments.

Dependencies:
- numpy>=1.21.0
- pandas>=1.3.0
- pyarrow>=6.0.0 (for parquet)

Usage:
    python generate_synthetic_bpjs.py --n_rows 100000 --fraud_ratio 0.03 --year 2024 --seed 42
"""

import numpy as np
import pandas as pd
import argparse
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BPJSDataGenerator:
    """Generate synthetic BPJS claims dataset with realistic fraud patterns."""
    
    # Reference data
    PROVINCES = ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara',
                 'Banten', 'Sulawesi Selatan', 'Kalimantan Timur', 'Bali', 'Riau']
    
    KABUPATEN = {
        'DKI Jakarta': ['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Barat', 'Jakarta Utara'],
        'Jawa Barat': ['Bandung', 'Bekasi', 'Bogor', 'Depok', 'Cirebon'],
        'Jawa Tengah': ['Semarang', 'Solo', 'Magelang', 'Pekalongan', 'Tegal'],
        'Jawa Timur': ['Surabaya', 'Malang', 'Kediri', 'Jember', 'Sidoarjo'],
        'Sumatera Utara': ['Medan', 'Deli Serdang', 'Binjai', 'Tebing Tinggi', 'Pematang Siantar']
    }
    
    ICD10_CODES = ['J00', 'J06.9', 'K29.7', 'I10', 'E11.9', 'M79.3', 'A09', 'J18.9', 'N39.0', 'K30']
    PROCEDURE_CODES = ['89.03', '93.39', '96.72', '87.44', '99.25', '88.72', '93.94', '87.03']
    
    FRAUD_TYPES = ['phantom', 'upcoding', 'unbundling', 'prolonged_los', 'identity', 'inflated_drugs', 'none']
    SEVERITY_LEVELS = ['ringan', 'sedang', 'berat', 'none']
    EVIDENCE_TYPES = ['system_anom', 'audit', 'whistleblower', 'none']
    
    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def make_providers(self, n_providers: int = 500) -> pd.DataFrame:
        """Generate provider/faskes pool with geographic distribution."""
        logger.info(f"Generating {n_providers} providers...")
        
        providers = []
        for i in range(n_providers):
            faskes_level = np.random.choice(['FKTP', 'FKRTL'], p=[0.7, 0.3])
            prov = np.random.choice(self.PROVINCES)
            kab_list = self.KABUPATEN.get(prov, ['Unknown'])
            kab = np.random.choice(kab_list)
            
            providers.append({
                'dpjp_id': f'D{i+1:05d}',
                'faskes_id': f'F{i+1:05d}',
                'faskes_level': faskes_level,
                'provinsi': prov,
                'kabupaten': kab,
                'kapitasi_rate': int(np.random.gamma(2, 25000)) if faskes_level == 'FKTP' else 0
            })
        
        return pd.DataFrame(providers)
    
    def make_participants(self, n_participants: int = 50000) -> pd.DataFrame:
        """Generate participant pool with realistic age distribution."""
        logger.info(f"Generating {n_participants} participants...")
        
        # Age follows a mixture: young (0-20), adult (20-60), elderly (60+)
        ages = np.concatenate([
            np.random.randint(0, 20, size=int(n_participants * 0.3)),
            np.random.randint(20, 60, size=int(n_participants * 0.5)),
            np.random.randint(60, 90, size=int(n_participants * 0.2))
        ])
        np.random.shuffle(ages)
        
        # Simulate NIK hash (some duplicates for identity fraud)
        nik_pool = [f'NIK{i:010d}' for i in range(int(n_participants * 0.98))]
        nik_duplicates = np.random.choice(nik_pool, size=int(n_participants * 0.02), replace=True)
        nik_hashes = np.concatenate([nik_pool, nik_duplicates])
        np.random.shuffle(nik_hashes)
        
        participants = pd.DataFrame({
            'participant_id': [f'P{i+1:08d}' for i in range(n_participants)],
            'nik_hash': nik_hashes[:n_participants],
            'age': ages[:n_participants],
            'sex': np.random.choice(['M', 'F'], size=n_participants)
        })
        
        return participants
    
    def sample_dates(self, n: int, year: int) -> np.ndarray:
        """Generate random dates within specified year."""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        days_range = (end_date - start_date).days
        
        random_days = np.random.randint(0, days_range + 1, size=n)
        dates = [start_date + timedelta(days=int(d)) for d in random_days]
        
        return np.array(dates)
    
    def gen_visit_counts(self, n: int, lam: float = 3.0, dispersion: float = 2.0) -> np.ndarray:
        """Generate visit counts using Negative Binomial (overdispersed) or Poisson."""
        if dispersion <= 0:
            return np.random.poisson(lam=lam, size=n)
        
        # NegBinomial parameterization: mean=lam, var=lam + dispersion
        mean = lam
        var = lam + dispersion
        r = (mean ** 2) / (var - mean) if var > mean else 1
        p = r / (r + mean)
        
        return np.random.negative_binomial(r, p, size=n)
    
    def gen_billed_amount(self, n: int, jenis_pelayanan: np.ndarray) -> np.ndarray:
        """Generate billed amounts using Gamma distribution with service-type variation."""
        # Rawat Jalan: lower amounts, Rawat Inap: higher amounts
        billed = np.zeros(n, dtype=int)
        
        rajal_mask = jenis_pelayanan == 'Rawat Jalan'
        ranap_mask = ~rajal_mask
        
        # Rawat Jalan: Gamma(shape=2, scale=150k) ~ mean 300k
        billed[rajal_mask] = np.round(
            np.random.gamma(2.0, 150_000, size=rajal_mask.sum())
        ).astype(int)
        
        # Rawat Inap: Gamma(shape=3, scale=500k) ~ mean 1.5M
        billed[ranap_mask] = np.round(
            np.random.gamma(3.0, 500_000, size=ranap_mask.sum())
        ).astype(int)
        
        return billed
    
    def gen_lama_dirawat(self, n: int, jenis_pelayanan: np.ndarray) -> np.ndarray:
        """Generate length of stay (LOS) for inpatient only."""
        los = np.zeros(n, dtype=int)
        ranap_mask = jenis_pelayanan == 'Rawat Inap'
        
        # Poisson with lambda=5 for realistic LOS
        los[ranap_mask] = np.random.poisson(lam=5, size=ranap_mask.sum()) + 1
        
        return los
    
    def assemble_claims(self, n_rows: int, providers_df: pd.DataFrame, 
                       participants_df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Assemble base claims dataset with realistic distributions."""
        logger.info(f"Assembling {n_rows} claims...")
        
        # Sample providers and participants
        provider_sample = providers_df.sample(n=n_rows, replace=True).reset_index(drop=True)
        participant_sample = participants_df.sample(n=n_rows, replace=True).reset_index(drop=True)
        
        # Service type distribution (70% outpatient, 30% inpatient)
        jenis_pelayanan = np.random.choice(
            ['Rawat Jalan', 'Rawat Inap'], 
            size=n_rows, 
            p=[0.7, 0.3]
        )
        
        # Generate monetary amounts
        billed_amount = self.gen_billed_amount(n_rows, jenis_pelayanan)
        
        # Drug and procedure costs as fractions of billed
        drug_fraction = np.random.beta(2, 8, size=n_rows)  # skewed toward lower values
        drug_cost = np.round(billed_amount * drug_fraction).astype(int)
        
        procedure_fraction = np.random.beta(3, 5, size=n_rows)
        procedure_cost = np.round(billed_amount * procedure_fraction * 0.6).astype(int)
        
        # Ensure drug + procedure <= billed
        total_cost = drug_cost + procedure_cost
        excess_mask = total_cost > billed_amount
        if excess_mask.any():
            scale_factor = billed_amount[excess_mask] / total_cost[excess_mask]
            drug_cost[excess_mask] = np.round(drug_cost[excess_mask] * scale_factor).astype(int)
            procedure_cost[excess_mask] = np.round(procedure_cost[excess_mask] * scale_factor).astype(int)
        
        # Payment ratio and paid amount
        payment_ratio = np.random.uniform(0.7, 1.0, size=n_rows)
        
        # Kapitasi cases: fixed payment
        kapitasi_flag = (provider_sample['faskes_level'] == 'FKTP') & (np.random.rand(n_rows) < 0.4)
        paid_amount = np.round(billed_amount * payment_ratio).astype(int)
        paid_amount[kapitasi_flag] = provider_sample.loc[kapitasi_flag, 'kapitasi_rate'].values
        paid_amount = np.minimum(paid_amount, billed_amount)  # paid cannot exceed billed
        
        # Generate other fields
        lama_dirawat = self.gen_lama_dirawat(n_rows, jenis_pelayanan)
        visit_count_30d = self.gen_visit_counts(n_rows, lam=2.5, dispersion=3.0)
        
        # Room class for inpatient only
        room_class = np.where(
            jenis_pelayanan == 'Rawat Inap',
            np.random.choice(['Kelas I', 'Kelas II', 'Kelas III', 'VIP'], size=n_rows),
            ''
        )
        
        # Referral patterns
        referral_flag = np.random.rand(n_rows) < 0.2
        referral_to_same = referral_flag & (np.random.rand(n_rows) < 0.15)  # suspicious pattern
        
        # Assemble dataframe
        df = pd.DataFrame({
            'claim_id': [f'C{i+1:08d}' for i in range(n_rows)],
            'episode_id': [f'E{i+1:08d}' for i in range(n_rows)],
            'participant_id': participant_sample['participant_id'].values,
            'nik_hash': participant_sample['nik_hash'].values,
            'age': participant_sample['age'].values,
            'faskes_id': provider_sample['faskes_id'].values,
            'faskes_level': provider_sample['faskes_level'].values,
            'provinsi': provider_sample['provinsi'].values,
            'kabupaten': provider_sample['kabupaten'].values,
            'tgl_pelayanan': self.sample_dates(n_rows, year),
            'kode_icd10': np.random.choice(self.ICD10_CODES, size=n_rows),
            'kode_prosedur': np.random.choice(self.PROCEDURE_CODES, size=n_rows),
            'jenis_pelayanan': jenis_pelayanan,
            'room_class': room_class,
            'lama_dirawat': lama_dirawat,
            'billed_amount': billed_amount,
            'paid_amount': paid_amount,
            'selisih_klaim': billed_amount - paid_amount,
            'dpjp_id': provider_sample['dpjp_id'].values,
            'kapitasi_flag': kapitasi_flag,
            'referral_flag': referral_flag,
            'drug_cost': drug_cost,
            'procedure_cost': procedure_cost,
            'visit_count_30d': visit_count_30d,
            'referral_to_same_facility': referral_to_same,
            'fraud_flag': 0,
            'fraud_type': 'none',
            'severity': 'none',
            'evidence_type': 'none'
        })
        
        return df
    
    def inject_fraud(self, df: pd.DataFrame, fraud_ratio: float = 0.03) -> pd.DataFrame:
        """Inject realistic fraud patterns into dataset - COMPLETE VERSION with 11 fraud types."""
        logger.info(f"Injecting fraud with ratio {fraud_ratio:.2%}...")
        
        n_total = len(df)
        n_fraud = int(n_total * fraud_ratio)
        
        # COMPLETE Fraud type distribution (11 types)
        fraud_mix = {
            'upcoding_diagnosis': 0.15,      # Upcoding Diagnosis
            'phantom_billing': 0.12,          # Phantom Billing (Klaim Fiktif)
            'cloning_claim': 0.08,            # Cloning Claim
            'inflated_bill': 0.12,            # Inflated Bill
            'service_unbundling': 0.10,       # Service Unbundling
            'self_referral': 0.10,            # Self-Referral
            'repeat_billing': 0.08,           # Repeat Billing
            'prolonged_los': 0.10,            # Prolonged Length of Stay
            'room_manipulation': 0.08,        # Manipulation of Room Charge
            'unnecessary_services': 0.05,     # Unnecessary Services
            'fake_license': 0.02              # Fake License / Illegal Practice
        }
        
        fraud_indices = np.random.choice(n_total, size=n_fraud, replace=False)
        type_counts = {k: int(n_fraud * v) for k, v in fraud_mix.items()}
        
        idx_pointer = 0
        
        # 1. UPCODING DIAGNOSIS (15%)
        n_upcode = type_counts['upcoding_diagnosis']
        upcode_idx = fraud_indices[idx_pointer:idx_pointer + n_upcode]
        df.loc[upcode_idx, 'billed_amount'] *= np.random.uniform(1.5, 2.5, size=len(upcode_idx))
        df.loc[upcode_idx, 'billed_amount'] = df.loc[upcode_idx, 'billed_amount'].astype(int)
        # Change ICD to more severe version
        df.loc[upcode_idx, 'kode_icd10'] = np.random.choice(['I10.9', 'E11.9', 'N18.9'], size=len(upcode_idx))
        df.loc[upcode_idx, 'fraud_flag'] = 1
        df.loc[upcode_idx, 'fraud_type'] = 'upcoding_diagnosis'
        df.loc[upcode_idx, 'severity'] = np.random.choice(['sedang', 'berat'], size=len(upcode_idx), p=[0.4, 0.6])
        df.loc[upcode_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_upcode
        
        # 2. PHANTOM BILLING - Klaim Fiktif (12%)
        n_phantom = type_counts['phantom_billing']
        phantom_idx = fraud_indices[idx_pointer:idx_pointer + n_phantom]
        df.loc[phantom_idx, 'billed_amount'] *= np.random.uniform(2.5, 6.0, size=len(phantom_idx))
        df.loc[phantom_idx, 'billed_amount'] = df.loc[phantom_idx, 'billed_amount'].astype(int)
        df.loc[phantom_idx, 'visit_count_30d'] = np.random.choice([0, 1], size=len(phantom_idx))
        df.loc[phantom_idx, 'fraud_flag'] = 1
        df.loc[phantom_idx, 'fraud_type'] = 'phantom_billing'
        df.loc[phantom_idx, 'severity'] = 'berat'
        df.loc[phantom_idx, 'evidence_type'] = np.random.choice(['system_anom', 'audit'], size=len(phantom_idx))
        idx_pointer += n_phantom
        
        # 3. CLONING CLAIM - Copy paste data pasien (8%)
        n_clone = type_counts['cloning_claim']
        clone_idx = fraud_indices[idx_pointer:idx_pointer + n_clone]
        # Simulate cloning by duplicating some patterns
        for idx in clone_idx:
            df.loc[idx, 'fraud_flag'] = 1
            df.loc[idx, 'fraud_type'] = 'cloning_claim'
            df.loc[idx, 'severity'] = 'berat'
            df.loc[idx, 'evidence_type'] = 'audit'
            # Same diagnosis and procedure as another random claim
            random_claim = df.sample(1).iloc[0]
            df.loc[idx, 'kode_icd10'] = random_claim['kode_icd10']
            df.loc[idx, 'kode_prosedur'] = random_claim['kode_prosedur']
        idx_pointer += n_clone
        
        # 4. INFLATED BILL - Gelembung tagihan obat/alkes (12%)
        n_inflated = type_counts['inflated_bill']
        inflated_idx = fraud_indices[idx_pointer:idx_pointer + n_inflated]
        df.loc[inflated_idx, 'drug_cost'] *= np.random.uniform(2.5, 8.0, size=len(inflated_idx))
        df.loc[inflated_idx, 'drug_cost'] = df.loc[inflated_idx, 'drug_cost'].astype(int)
        df.loc[inflated_idx, 'procedure_cost'] *= np.random.uniform(1.5, 3.0, size=len(inflated_idx))
        df.loc[inflated_idx, 'procedure_cost'] = df.loc[inflated_idx, 'procedure_cost'].astype(int)
        df.loc[inflated_idx, 'billed_amount'] = (
            df.loc[inflated_idx, 'drug_cost'] + df.loc[inflated_idx, 'procedure_cost']
        ).astype(int)
        df.loc[inflated_idx, 'fraud_flag'] = 1
        df.loc[inflated_idx, 'fraud_type'] = 'inflated_bill'
        df.loc[inflated_idx, 'severity'] = 'sedang'
        df.loc[inflated_idx, 'evidence_type'] = 'system_anom'
        idx_pointer += n_inflated
        
        # 5. SERVICE UNBUNDLING - Pecah paket jadi multiple claims (10%)
        n_unbundle = type_counts['service_unbundling']
        unbundle_idx = fraud_indices[idx_pointer:idx_pointer + n_unbundle]
        df.loc[unbundle_idx, 'procedure_cost'] *= np.random.uniform(2.0, 4.0, size=len(unbundle_idx))
        df.loc[unbundle_idx, 'billed_amount'] = (
            df.loc[unbundle_idx, 'drug_cost'] + df.loc[unbundle_idx, 'procedure_cost']
        ).astype(int)
        df.loc[unbundle_idx, 'fraud_flag'] = 1
        df.loc[unbundle_idx, 'fraud_type'] = 'service_unbundling'
        df.loc[unbundle_idx, 'severity'] = 'sedang'
        df.loc[unbundle_idx, 'evidence_type'] = 'system_anom'
        idx_pointer += n_unbundle
        
        # 6. SELF-REFERRAL - Rujuk ke faskes milik sendiri (10%)
        n_self_ref = type_counts['self_referral']
        self_ref_idx = fraud_indices[idx_pointer:idx_pointer + n_self_ref]
        df.loc[self_ref_idx, 'referral_flag'] = True
        df.loc[self_ref_idx, 'referral_to_same_facility'] = True
        df.loc[self_ref_idx, 'billed_amount'] *= np.random.uniform(1.3, 2.0, size=len(self_ref_idx))
        df.loc[self_ref_idx, 'billed_amount'] = df.loc[self_ref_idx, 'billed_amount'].astype(int)
        df.loc[self_ref_idx, 'fraud_flag'] = 1
        df.loc[self_ref_idx, 'fraud_type'] = 'self_referral'
        df.loc[self_ref_idx, 'severity'] = 'sedang'
        df.loc[self_ref_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_self_ref
        
        # 7. REPEAT BILLING - Klaim yang sama ditagih 2x (8%)
        n_repeat = type_counts['repeat_billing']
        repeat_idx = fraud_indices[idx_pointer:idx_pointer + n_repeat]
        # Mark as repeat billing
        df.loc[repeat_idx, 'fraud_flag'] = 1
        df.loc[repeat_idx, 'fraud_type'] = 'repeat_billing'
        df.loc[repeat_idx, 'severity'] = 'berat'
        df.loc[repeat_idx, 'evidence_type'] = 'system_anom'
        idx_pointer += n_repeat
        
        # 8. PROLONGED LOS - Perpanjang rawat inap tanpa indikasi (10%)
        n_prolong = type_counts['prolonged_los']
        prolong_idx = fraud_indices[idx_pointer:idx_pointer + n_prolong]
        
        ranap_mask = df.loc[prolong_idx, 'jenis_pelayanan'] == 'Rawat Inap'
        ranap_prolong = prolong_idx[ranap_mask]
        rajal_prolong = prolong_idx[~ranap_mask]
        
        if len(ranap_prolong) > 0:
            df.loc[ranap_prolong, 'lama_dirawat'] = np.random.randint(15, 60, size=len(ranap_prolong))
            df.loc[ranap_prolong, 'billed_amount'] *= np.random.uniform(2.0, 4.0, size=len(ranap_prolong))
            df.loc[ranap_prolong, 'billed_amount'] = df.loc[ranap_prolong, 'billed_amount'].astype(int)
            df.loc[ranap_prolong, 'fraud_flag'] = 1
            df.loc[ranap_prolong, 'fraud_type'] = 'prolonged_los'
            df.loc[ranap_prolong, 'severity'] = 'sedang'
            df.loc[ranap_prolong, 'evidence_type'] = 'audit'
        
        # Convert rawat jalan yang terpilih ke upcoding
        if len(rajal_prolong) > 0:
            df.loc[rajal_prolong, 'billed_amount'] *= np.random.uniform(1.5, 2.5, size=len(rajal_prolong))
            df.loc[rajal_prolong, 'billed_amount'] = df.loc[rajal_prolong, 'billed_amount'].astype(int)
            df.loc[rajal_prolong, 'fraud_flag'] = 1
            df.loc[rajal_prolong, 'fraud_type'] = 'upcoding_diagnosis'
            df.loc[rajal_prolong, 'severity'] = 'sedang'
            df.loc[rajal_prolong, 'evidence_type'] = 'audit'
        
        idx_pointer += n_prolong
        
        # 9. ROOM MANIPULATION - Manipulasi kelas perawatan (8%)
        n_room = type_counts['room_manipulation']
        room_idx = fraud_indices[idx_pointer:idx_pointer + n_room]
        
        ranap_mask = df.loc[room_idx, 'jenis_pelayanan'] == 'Rawat Inap'
        ranap_room = room_idx[ranap_mask]
        
        if len(ranap_room) > 0:
            # Claim kelas lebih tinggi dari sebenarnya
            df.loc[ranap_room, 'room_class'] = np.random.choice(['Kelas I', 'VIP'], size=len(ranap_room))
            df.loc[ranap_room, 'billed_amount'] *= np.random.uniform(1.4, 2.2, size=len(ranap_room))
            df.loc[ranap_room, 'billed_amount'] = df.loc[ranap_room, 'billed_amount'].astype(int)
            df.loc[ranap_room, 'fraud_flag'] = 1
            df.loc[ranap_room, 'fraud_type'] = 'room_manipulation'
            df.loc[ranap_room, 'severity'] = 'sedang'
            df.loc[ranap_room, 'evidence_type'] = 'audit'
        
        # Rawat jalan yang terpilih, convert ke inflated bill
        rajal_room = room_idx[~ranap_mask]
        if len(rajal_room) > 0:
            df.loc[rajal_room, 'drug_cost'] *= np.random.uniform(2, 5, size=len(rajal_room))
            df.loc[rajal_room, 'drug_cost'] = df.loc[rajal_room, 'drug_cost'].astype(int)
            df.loc[rajal_room, 'billed_amount'] *= np.random.uniform(1.5, 2.5, size=len(rajal_room))
            df.loc[rajal_room, 'billed_amount'] = df.loc[rajal_room, 'billed_amount'].astype(int)
            df.loc[rajal_room, 'fraud_flag'] = 1
            df.loc[rajal_room, 'fraud_type'] = 'inflated_bill'
            df.loc[rajal_room, 'severity'] = 'sedang'
            df.loc[rajal_room, 'evidence_type'] = 'system_anom'
        
        idx_pointer += n_room
        
        # 10. UNNECESSARY SERVICES - Layanan berlebihan (5%)
        n_unnecessary = type_counts['unnecessary_services']
        unnecessary_idx = fraud_indices[idx_pointer:idx_pointer + n_unnecessary]
        df.loc[unnecessary_idx, 'procedure_cost'] *= np.random.uniform(1.5, 2.5, size=len(unnecessary_idx))
        df.loc[unnecessary_idx, 'visit_count_30d'] = np.random.randint(5, 15, size=len(unnecessary_idx))
        df.loc[unnecessary_idx, 'billed_amount'] = (
            df.loc[unnecessary_idx, 'drug_cost'] + df.loc[unnecessary_idx, 'procedure_cost']
        ).astype(int)
        df.loc[unnecessary_idx, 'fraud_flag'] = 1
        df.loc[unnecessary_idx, 'fraud_type'] = 'unnecessary_services'
        df.loc[unnecessary_idx, 'severity'] = np.random.choice(['ringan', 'sedang'], size=len(unnecessary_idx))
        df.loc[unnecessary_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_unnecessary
        
        # 11. FAKE LICENSE - Izin praktik palsu (2%)
        n_fake = type_counts['fake_license']
        fake_idx = fraud_indices[idx_pointer:idx_pointer + n_fake]
        df.loc[fake_idx, 'billed_amount'] *= np.random.uniform(1.2, 2.0, size=len(fake_idx))
        df.loc[fake_idx, 'billed_amount'] = df.loc[fake_idx, 'billed_amount'].astype(int)
        df.loc[fake_idx, 'fraud_flag'] = 1
        df.loc[fake_idx, 'fraud_type'] = 'fake_license'
        df.loc[fake_idx, 'severity'] = 'berat'
        df.loc[fake_idx, 'evidence_type'] = 'whistleblower'
        
        # Recalculate paid_amount and selisih after fraud injection
        df['paid_amount'] = np.minimum(df['paid_amount'], df['billed_amount'])
        df['selisih_klaim'] = df['billed_amount'] - df['paid_amount']
        
        # Summary by fraud type
        fraud_summary = df[df['fraud_flag'] == 1]['fraud_type'].value_counts().sort_index()
        logger.info(f"Fraud injection complete: {df['fraud_flag'].sum()} fraudulent claims")
        logger.info("Fraud distribution by type:")
        for fraud_type, count in fraud_summary.items():
            percentage = count / df['fraud_flag'].sum() * 100
            logger.info(f"  - {fraud_type}: {count} ({percentage:.1f}%)")
        
        return df
    
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for ML."""
        logger.info("Computing engineered features...")
        
        # Claim ratio
        df['claim_ratio'] = df['billed_amount'] / (df['paid_amount'] + 1)
        
        # Drug ratio
        df['drug_ratio'] = df['drug_cost'] / (df['billed_amount'] + 1)
        
        # Procedure ratio
        df['procedure_ratio'] = df['procedure_cost'] / (df['billed_amount'] + 1)
        
        # Provider claim share (monthly)
        df['year_month'] = pd.to_datetime(df['tgl_pelayanan']).dt.to_period('M')
        provider_monthly_counts = df.groupby(['dpjp_id', 'year_month']).size().reset_index(name='provider_claims')
        monthly_total = df.groupby('year_month').size().reset_index(name='total_claims')
        
        provider_stats = provider_monthly_counts.merge(monthly_total, on='year_month')
        provider_stats['provider_claim_share'] = provider_stats['provider_claims'] / provider_stats['total_claims']
        
        df = df.merge(
            provider_stats[['dpjp_id', 'year_month', 'provider_claim_share']], 
            on=['dpjp_id', 'year_month'], 
            how='left'
        )
        df['provider_claim_share'] = df['provider_claim_share'].fillna(0)
        
        df.drop('year_month', axis=1, inplace=True)
        
        return df


def validate(df: pd.DataFrame, fraud_ratio: float) -> bool:
    """Run validation checks on generated dataset."""
    logger.info("Running validation checks...")
    
    checks_passed = True
    
    # Check 1: No negative amounts
    if (df[['billed_amount', 'paid_amount', 'drug_cost', 'procedure_cost']] < 0).any().any():
        logger.error("❌ FAIL: Found negative monetary values")
        checks_passed = False
    else:
        logger.info("✓ PASS: No negative amounts")
    
    # Check 2: paid <= billed
    if not (df['paid_amount'] <= df['billed_amount']).all():
        logger.error("❌ FAIL: Found paid_amount > billed_amount")
        checks_passed = False
    else:
        logger.info("✓ PASS: paid_amount <= billed_amount")
    
    # Check 3: Fraud ratio within tolerance
    actual_fraud_ratio = df['fraud_flag'].mean()
    tolerance = fraud_ratio * 0.2
    if abs(actual_fraud_ratio - fraud_ratio) > tolerance:
        logger.warning(f"⚠ WARNING: Fraud ratio {actual_fraud_ratio:.2%} outside tolerance of {fraud_ratio:.2%} ± {tolerance:.2%}")
    else:
        logger.info(f"✓ PASS: Fraud ratio {actual_fraud_ratio:.2%} within tolerance")
    
    # Check 4: drug + procedure reasonable relative to billed
    df['cost_sum'] = df['drug_cost'] + df['procedure_cost']
    if (df['cost_sum'] > df['billed_amount'] * 1.1).any():
        logger.warning("⚠ WARNING: Some claims have drug+procedure > 110% of billed (possible fraud)")
    else:
        logger.info("✓ PASS: Component costs reasonable")
    
    return checks_passed


def save_outputs(df: pd.DataFrame, out_dir: str, params: Dict):
    """Save dataset and metadata to files."""
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, 'synthetic_bpjs_claims.csv')
    parquet_path = os.path.join(out_dir, 'synthetic_bpjs_claims.parquet')
    meta_path = os.path.join(out_dir, 'metadata.json')
    
    logger.info(f"Saving CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saving Parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False, compression='snappy')
    
    # Save metadata
    metadata = {
        'generation_params': params,
        'timestamp': datetime.now().isoformat(),
        'n_rows': len(df),
        'n_fraud': int(df['fraud_flag'].sum()),
        'fraud_ratio_actual': float(df['fraud_flag'].mean()),
        'columns': list(df.columns)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {meta_path}")


def print_summary(df: pd.DataFrame):
    """Print dataset summary and ML usage tips."""
    print("\n" + "=" * 70)
    print("BPJS SYNTHETIC DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Claims: {len(df):,}")
    print(f"Fraudulent Claims: {df['fraud_flag'].sum():,} ({df['fraud_flag'].mean():.2%})")
    
    print("\n--- Fraud Type Distribution ---")
    print(df[df['fraud_flag'] == 1]['fraud_type'].value_counts())
    
    print("\n--- Monetary Statistics (IDR) ---")
    print(df[['billed_amount', 'paid_amount', 'drug_cost']].describe().round(0))
    
    print("\n--- Sample Records ---")
    print(df.head(10).to_string())
    
    print("\n" + "=" * 70)
    print("ML USAGE TIPS")
    print("=" * 70)
    print("""
1. Train/Test Split: Use time-based split (e.g., last 2 months as test set)
2. Stratification: Stratify by 'severity' to ensure balanced fraud types
3. Recommended Metrics:
   - AUPRC (Area Under Precision-Recall Curve)
   - Precision@k (k=100, 500, 1000)
   - Cost-savings simulation
   - Time-to-detect
4. Feature Engineering:
   - Use computed features: claim_ratio, drug_ratio, provider_claim_share
   - Consider temporal features from tgl_pelayanan
   - Aggregate provider-level statistics
5. Class Imbalance: Consider SMOTE, class weights, or cost-sensitive learning
6. Holdout Set: Reserve high-severity fraud cases for final evaluation
    """)
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic BPJS claims dataset')
    parser.add_argument('--n_rows', type=int, default=100_000, help='Number of claim records')
    parser.add_argument('--fraud_ratio', type=float, default=0.03, help='Proportion of fraudulent claims')
    parser.add_argument('--year', type=int, default=2024, help='Year for claim dates')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--n_providers', type=int, default=500, help='Number of providers')
    parser.add_argument('--n_participants', type=int, default=50000, help='Number of participants')
    
    args = parser.parse_args()
    
    logger.info("Starting BPJS synthetic dataset generation...")
    logger.info(f"Parameters: n_rows={args.n_rows}, fraud_ratio={args.fraud_ratio}, year={args.year}, seed={args.seed}")
    
    # Initialize generator
    generator = BPJSDataGenerator(seed=args.seed)
    
    # Generate providers and participants
    providers_df = generator.make_providers(n_providers=args.n_providers)
    participants_df = generator.make_participants(n_participants=args.n_participants)
    
    # Assemble base claims
    df = generator.assemble_claims(args.n_rows, providers_df, participants_df, args.year)
    
    # Inject fraud patterns
    df = generator.inject_fraud(df, fraud_ratio=args.fraud_ratio)
    
    # Add engineered features
    df = generator.featurize(df)
    
    # Validate dataset
    validate(df, args.fraud_ratio)
    
    # Save outputs
    params = vars(args)
    save_outputs(df, args.out_dir, params)
    
    # Print summary
    print_summary(df)
    
    logger.info("✅ Dataset generation complete!")


