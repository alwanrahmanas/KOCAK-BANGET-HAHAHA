"""
BPJS Fraud Detection Dataset Generator V3.2 - ENHANCED PHANTOM BILLING
========================================================================
✅ NEW: 5 Previous Claims History per Patient
✅ NEW: Drug Dispensing Logic with Match Score
✅ NEW: Enhanced Clinical Deviation Scoring
✅ NEW: Multi-Pattern Phantom Billing Injection
✅ NEW: Red Flags Aggregate Features
✅ KEPT: All existing columns + new enhancements

Version: 3.2-FINAL-PHANTOM
Last Updated: 2024-11
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import hashlib
import argparse
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========== ENHANCED MAPPING TABLES ==========

ICD10_TO_NAME = {
    'J00': 'Common Cold',
    'I10': 'Hypertension',
    'E11.9': 'Diabetes Type 2',
    'J18.9': 'Pneumonia',
    'N39.0': 'UTI',
    'K29.7': 'Gastritis',
    'M-4-10-I': 'Femur Fracture'
}

PROCEDURE_TO_NAME = {
    '89.03': 'Physical Examination',
    '99.25': 'Blood Pressure Monitoring',
    '96.72': 'Blood Glucose Test',
    '87.44': 'Chest X-Ray',
    '93.39': 'Respiratory Therapy',
    '88.72': 'Urine Analysis',
    '87.03': 'Bone X-Ray',
    '93.94': 'Physical Therapy'
}

# ========== NEW: DRUG LOOKUP TABLE ==========

DRUG_LOOKUP = {
    'J00': ['paracetamol', 'cetirizine', 'vitamin_c', 'dexamethasone'],
    'I10': ['amlodipine', 'captopril', 'hydrochlorothiazide', 'losartan'],
    'E11.9': ['metformin', 'glimepiride', 'insulin', 'glibenclamide'],
    'J18.9': ['ceftriaxone', 'salbutamol', 'ambroxol', 'oxygen_therapy', 'azithromycin'],
    'N39.0': ['ciprofloxacin', 'cefadroxil', 'nitrofurantoin', 'cotrimoxazole'],
    'K29.7': ['omeprazole', 'ranitidine', 'sucralfate', 'antacid'],
    'M-4-10-I': ['ketorolac', 'tramadol', 'cefazolin', 'heparin', 'calcium_supplement']
}

# Drug mismatches for phantom billing scenarios
PHANTOM_MISMATCH_DRUGS = {
    'J00': ['insulin', 'furosemide', 'heparin'],  # Cold with diabetes/cardiac drugs
    'K29.7': ['insulin', 'ceftriaxone', 'chemotherapy'],  # Gastritis with severe drugs
    'I10': ['chemotherapy', 'immunosuppressant'],  # Hypertension with cancer drugs
}


# ========== HELPER CLASSES ==========

class ClinicalPathwayManager:
    """Manage clinical pathways and INA-CBG tariff references."""

    PATHWAYS = {
        'J00': {
            'name': 'Nasofaringitis Akut (Common Cold)',
            'service_type': 'Rawat Jalan',
            'typical_los': 0,
            'typical_procedures': ['89.03'],
            'cost_range': (100_000, 500_000),
            'inacbg_code': 'M-1-20-I',
            'tarif_kelas_3': 383_300,
            'tarif_kelas_2': 447_100,
            'tarif_kelas_1': 510_900
        },
        'I10': {
            'name': 'Hipertensi Esensial',
            'service_type': 'Rawat Jalan',
            'typical_los': 0,
            'typical_procedures': ['89.03', '99.25'],
            'cost_range': (200_000, 800_000),
            'inacbg_code': 'M-1-30-I',
            'tarif_kelas_3': 605_400,
            'tarif_kelas_2': 706_100,
            'tarif_kelas_1': 806_800
        },
        'E11.9': {
            'name': 'Diabetes Melitus Tipe 2',
            'service_type': 'Rawat Jalan',
            'typical_los': 0,
            'typical_procedures': ['89.03', '96.72'],
            'cost_range': (300_000, 1_200_000),
            'inacbg_code': 'M-1-40-I',
            'tarif_kelas_3': 906_900,
            'tarif_kelas_2': 1_057_900,
            'tarif_kelas_1': 1_208_900
        },
        'J18.9': {
            'name': 'Pneumonia',
            'service_type': 'Rawat Inap',
            'typical_los': 5,
            'typical_procedures': ['87.44', '96.72', '93.39'],
            'cost_range': (3_000_000, 8_000_000),
            'inacbg_code': 'M-1-50-I',
            'tarif_kelas_3': 5_448_800,
            'tarif_kelas_2': 6_347_900,
            'tarif_kelas_1': 7_246_900
        },
        'N39.0': {
            'name': 'Infeksi Saluran Kemih',
            'service_type': 'Rawat Jalan',
            'typical_los': 0,
            'typical_procedures': ['89.03', '88.72'],
            'cost_range': (200_000, 700_000),
            'inacbg_code': 'M-1-70-I',
            'tarif_kelas_3': 590_500,
            'tarif_kelas_2': 688_700,
            'tarif_kelas_1': 786_900
        },
        'K29.7': {
            'name': 'Gastritis',
            'service_type': 'Rawat Jalan',
            'typical_los': 0,
            'typical_procedures': ['89.03'],
            'cost_range': (150_000, 600_000),
            'inacbg_code': 'M-1-60-I',
            'tarif_kelas_3': 426_100,
            'tarif_kelas_2': 497_000,
            'tarif_kelas_1': 567_900
        },
        'M-4-10-I': {
            'name': 'Fraktur Femur',
            'service_type': 'Rawat Inap',
            'typical_los': 7,
            'typical_procedures': ['87.03', '93.94'],
            'cost_range': (5_000_000, 15_000_000),
            'inacbg_code': 'M-4-10-I',
            'tarif_kelas_3': 3_124_600,
            'tarif_kelas_2': 3_756_700,
            'tarif_kelas_1': 4_288_700
        }
    }

    @classmethod
    def get_pathway(cls, icd10_code: str) -> Dict:
        """Get clinical pathway for ICD-10 code."""
        return cls.PATHWAYS.get(icd10_code, cls.PATHWAYS['J00'])

    @classmethod
    def calculate_deviation_score_enhanced(cls, row: pd.Series) -> float:
        """
        MODUL C: Enhanced clinical pathway deviation score with penalties.
        Range: 0.0 - 3.0+ (higher = more deviation)
        """
        pathway = cls.get_pathway(row['kode_icd10'])
        deviation_factors = []

        # Length of stay deviation
        if pathway['typical_los'] > 0:
            if row['lama_dirawat'] == 0:
                deviation_factors.append(1.0)  # Strong penalty
            else:
                los_deviation = abs(row['lama_dirawat'] - pathway['typical_los']) / pathway['typical_los']
                deviation_factors.append(min(los_deviation, 1.0))

        # Cost deviation
        expected_cost = (pathway['cost_range'][0] + pathway['cost_range'][1]) / 2
        cost_deviation = abs(row['billed_amount'] - expected_cost) / expected_cost
        deviation_factors.append(min(cost_deviation, 1.0))

        # Service type mismatch
        if row['jenis_pelayanan'] != pathway['service_type']:
            deviation_factors.append(1.0)

        # Procedure mismatch
        if row['kode_prosedur'] not in pathway['typical_procedures']:
            deviation_factors.append(1.0)

        # LOS = 0 but should be inpatient
        if row['lama_dirawat'] == 0 and pathway['typical_los'] > 1:
            deviation_factors.append(1.0)

        return np.mean(deviation_factors) if deviation_factors else 0.0


class DrugDispenseManager:
    """MODUL B: Manage drug dispensing and matching."""

    @staticmethod
    def generate_drug_list(icd10_code: str, is_fraud: bool = False,
                          fraud_type: str = 'none') -> List[str]:
        """Generate realistic drug list based on diagnosis."""
        if is_fraud and fraud_type == 'phantom_billing':
            # Phantom billing: mismatch drugs
            if icd10_code in PHANTOM_MISMATCH_DRUGS:
                mismatch_drugs = PHANTOM_MISMATCH_DRUGS[icd10_code]
                return np.random.choice(mismatch_drugs,
                                       size=min(2, len(mismatch_drugs)),
                                       replace=False).tolist()

        # Normal case: appropriate drugs
        appropriate_drugs = DRUG_LOOKUP.get(icd10_code, ['paracetamol'])
        n_drugs = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        n_drugs = min(n_drugs, len(appropriate_drugs))

        return np.random.choice(appropriate_drugs, size=n_drugs, replace=False).tolist()

    @staticmethod
    def calculate_drug_match_score(dispensed_drugs: List[str], icd10_code: str) -> float:
        """
        Calculate Jaccard similarity between dispensed drugs and expected drugs.
        Range: 0.0 - 1.0
        """
        if not dispensed_drugs:
            return 0.0

        expected_drugs = set(DRUG_LOOKUP.get(icd10_code, []))
        dispensed_set = set(dispensed_drugs)

        if not expected_drugs:
            return 0.5  # Neutral if no expected drugs

        intersection = len(dispensed_set & expected_drugs)
        union = len(dispensed_set | expected_drugs)

        return intersection / union if union > 0 else 0.0


class HistoricalClaimsManager:
    """MODUL A: Manage 5 previous claims history per patient."""

    @staticmethod
    def extract_prev_claims_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from 5 previous claims for each patient.
        Returns enhanced dataframe with prev_claim_1 to prev_claim_5 columns.
        """
        logger.info("Extracting 5 previous claims history per patient...")

        # Sort by patient and date
        df = df.sort_values(['participant_id', 'tgl_pelayanan']).reset_index(drop=True)

        # Initialize columns for 5 previous claims
        prev_cols_template = [
            'kode_icd10', 'jenis_pelayanan', 'room_class', 'billed_amount',
            'claim_ratio', 'clinical_pathway_deviation_score', 'visit_count_30d', 'faskes_id'
        ]

        for i in range(1, 6):
            for col in prev_cols_template:
                df[f'prev_claim_{i}_{col}'] = None

        # Group by patient and extract history
        for participant_id, group in df.groupby('participant_id'):
            indices = group.index.tolist()

            for idx_pos, idx in enumerate(indices):
                # Get previous 5 claims
                prev_indices = indices[max(0, idx_pos-5):idx_pos]

                for i, prev_idx in enumerate(reversed(prev_indices)):
                    claim_num = i + 1
                    if claim_num > 5:
                        break

                    for col in prev_cols_template:
                        if col in df.columns:
                            df.at[idx, f'prev_claim_{claim_num}_{col}'] = df.at[prev_idx, col]

        # Calculate aggregate features
        df = HistoricalClaimsManager._calculate_aggregate_features(df)

        logger.info("✅ Previous claims history extracted")
        return df

    @staticmethod
    def _calculate_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate statistics from 5 previous claims."""

        # Count valid previous claims
        df['total_klaim_5x'] = 0
        for i in range(1, 6):
            df['total_klaim_5x'] += df[f'prev_claim_{i}_billed_amount'].notna().astype(int)

        # Average billed amount
        billed_cols = [f'prev_claim_{i}_billed_amount' for i in range(1, 6)]
        df['rerata_billed_5x'] = df[billed_cols].mean(axis=1, skipna=True)

        # Std of claim ratio
        ratio_cols = [f'prev_claim_{i}_claim_ratio' for i in range(1, 6)]
        df['std_claim_ratio_5x'] = df[ratio_cols].std(axis=1, skipna=True)

        # Average LOS (need to calculate from service type)
        df['rerata_lama_dirawat_5x'] = 0.0  # Will be calculated if LOS data available

        # Unique facilities visited
        df['total_rs_unique_visited_5x'] = df.apply(
            lambda row: len(set([row[f'prev_claim_{i}_faskes_id']
                                for i in range(1, 6)
                                if pd.notna(row[f'prev_claim_{i}_faskes_id'])])),
            axis=1
        )

        # Unique diagnoses
        df['total_diagnosis_unique_5x'] = df.apply(
            lambda row: len(set([row[f'prev_claim_{i}_kode_icd10']
                                for i in range(1, 6)
                                if pd.notna(row[f'prev_claim_{i}_kode_icd10'])])),
            axis=1
        )

        # Fill NaN aggregates with 0
        aggregate_cols = ['rerata_billed_5x', 'std_claim_ratio_5x', 'rerata_lama_dirawat_5x']
        df[aggregate_cols] = df[aggregate_cols].fillna(0)

        return df


class RedFlagsCalculator:
    """MODUL E: Calculate red flags aggregate scores."""

    @staticmethod
    def calculate_red_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Add red flag indicators and composite score."""
        logger.info("Calculating red flags...")

        # Individual flags
        df['visit_suspicious_flag'] = (df['visit_count_30d'] == 0).astype(int)
        df['obat_mismatch_flag'] = (df['obat_match_score'] < 0.2).astype(int)

        # Billing spike flag
        df['billing_spike_flag'] = 0
        mask = df['rerata_billed_5x'] > 0
        df.loc[mask, 'billing_spike_flag'] = (
            df.loc[mask, 'billed_amount'] > 3 * df.loc[mask, 'rerata_billed_5x']
        ).astype(int)

        # High deviation flag
        df['high_deviation_flag'] = (df['clinical_pathway_deviation_score'] > 0.7).astype(int)

        # Composite phantom suspect score
        df['phantom_suspect_score'] = (
            df['visit_suspicious_flag'] * 0.3 +
            df['obat_mismatch_flag'] * 0.3 +
            df['billing_spike_flag'] * 0.2 +
            df['high_deviation_flag'] * 0.2
        )

        logger.info("✅ Red flags calculated")
        return df


class GraphFraudInjector:
    """Inject graph-based fraud patterns."""

    @staticmethod
    def inject_ring_network(df: pd.DataFrame, n_rings: int = 3, ring_size: int = 5) -> pd.DataFrame:
        """Inject circular referral chains."""
        logger.info(f"Injecting {n_rings} ring networks (size={ring_size})...")
        available_idx = df[df['fraud_flag'] == 0].index.tolist()

        for ring_id in range(n_rings):
            if len(available_idx) < ring_size:
                break

            ring_indices = np.random.choice(available_idx, size=ring_size, replace=False)
            available_idx = [i for i in available_idx if i not in ring_indices]
            providers = [f'RING{ring_id}_P{i}' for i in range(ring_size)]

            for i, idx in enumerate(ring_indices):
                current_provider = providers[i]
                next_provider = providers[(i + 1) % ring_size]

                df.loc[idx, 'dpjp_id'] = current_provider
                df.loc[idx, 'referral_flag'] = True
                df.loc[idx, 'referral_to'] = next_provider
                df.loc[idx, 'fraud_flag'] = 1
                df.loc[idx, 'fraud_type'] = 'graph_ring_network'
                df.loc[idx, 'severity'] = 'berat'
                df.loc[idx, 'evidence_type'] = 'graph_analysis'
                df.loc[idx, 'graph_pattern_id'] = f'RING_{ring_id}'
                df.loc[idx, 'billed_amount'] = int(df.loc[idx, 'billed_amount'] * np.random.uniform(1.5, 2.5))

        logger.info(f"✅ Injected {n_rings} ring networks")
        return df

    @staticmethod
    def inject_star_pattern(df: pd.DataFrame, n_stars: int = 5, spokes_per_star: int = 8) -> pd.DataFrame:
        """Inject star patterns (hub-and-spoke)."""
        logger.info(f"Injecting {n_stars} star patterns...")
        available_idx = df[df['fraud_flag'] == 0].index.tolist()

        for star_id in range(n_stars):
            needed = spokes_per_star + 1
            if len(available_idx) < needed:
                break

            star_indices = np.random.choice(available_idx, size=needed, replace=False)
            available_idx = [i for i in available_idx if i not in star_indices]

            hub_idx = star_indices[0]
            spoke_indices = star_indices[1:]
            hub_provider = f'STAR{star_id}_HUB'

            df.loc[hub_idx, 'dpjp_id'] = hub_provider
            df.loc[hub_idx, 'fraud_flag'] = 1
            df.loc[hub_idx, 'fraud_type'] = 'graph_star_center'
            df.loc[hub_idx, 'severity'] = 'berat'
            df.loc[hub_idx, 'graph_pattern_id'] = f'STAR_{star_id}'

            for i, spoke_idx in enumerate(spoke_indices):
                df.loc[spoke_idx, 'dpjp_id'] = f'STAR{star_id}_S{i}'
                df.loc[spoke_idx, 'referral_to'] = hub_provider
                df.loc[spoke_idx, 'fraud_flag'] = 1
                df.loc[spoke_idx, 'fraud_type'] = 'graph_star_spoke'

        return df

    @staticmethod
    def inject_clique_collusion(df: pd.DataFrame, n_cliques: int = 3, clique_size: int = 4) -> pd.DataFrame:
        """Inject fully-connected fraud groups (cliques)."""
        logger.info(f"Injecting {n_cliques} clique collusions (size={clique_size})...")

        available_idx = df[df['fraud_flag'] == 0].index.tolist()

        for clique_id in range(n_cliques):
            if len(available_idx) < clique_size:
                break

            clique_indices = np.random.choice(available_idx, size=clique_size, replace=False)
            available_idx = [i for i in available_idx if i not in clique_indices]
            providers = [f'CLIQUE{clique_id}_P{i}' for i in range(clique_size)]

            for i, idx in enumerate(clique_indices):
                current_provider = providers[i]
                other_providers = [p for p in providers if p != current_provider]
                refer_to = np.random.choice(other_providers)

                df.loc[idx, 'dpjp_id'] = current_provider
                df.loc[idx, 'referral_flag'] = True
                df.loc[idx, 'referral_to'] = refer_to
                df.loc[idx, 'fraud_flag'] = 1
                df.loc[idx, 'fraud_type'] = 'graph_clique_collusion'
                df.loc[idx, 'severity'] = 'berat'
                df.loc[idx, 'evidence_type'] = 'graph_analysis'
                df.loc[idx, 'graph_pattern_id'] = f'CLIQUE_{clique_id}'
                df.loc[idx, 'billed_amount'] = int(df.loc[idx, 'billed_amount'] * np.random.uniform(1.8, 3.0))

        logger.info(f"✅ Injected {n_cliques} clique collusions")
        return df

    @classmethod
    def inject_all_graph_patterns(cls, df: pd.DataFrame, graph_fraud_ratio: float = 0.02) -> pd.DataFrame:
        """Inject all graph fraud patterns."""
        logger.info(f"Injecting graph fraud patterns ({graph_fraud_ratio:.2%})...")

        if 'referral_to' not in df.columns:
            df['referral_to'] = ''
        if 'graph_pattern_id' not in df.columns:
            df['graph_pattern_id'] = ''
        # logger.info(f"Target graph fraud cases: {n_graph_fraud:,} ({graph_fraud_ratio:.2%})")

        n_graph_fraud = int(len(df) * graph_fraud_ratio)
        n_rings = max(1, int(n_graph_fraud * 0.4 / 5))
        n_stars = max(1, int(n_graph_fraud * 0.6 / 9))
        n_cliques = max(1, int(n_graph_fraud * 0.15 / 4))
        df = cls.inject_clique_collusion(df, n_cliques=n_cliques, clique_size=4)
        df = cls.inject_ring_network(df, n_rings=n_rings)
        df = cls.inject_star_pattern(df, n_stars=n_stars)

        logger.info(f"✅ Graph fraud injection complete")
        return df


# ========== MAIN GENERATOR CLASS ==========

class BPJSDataGeneratorV32:
    """Enhanced BPJS dataset generator with phantom billing focus."""

    PROVINCE_DISTRIBUTION = {
        'Jawa Barat': 0.20, 'Jawa Timur': 0.18, 'Jawa Tengah': 0.15,
        'DKI Jakarta': 0.12, 'Sumatera Utara': 0.10, 'Banten': 0.08,
        'Sulawesi Selatan': 0.07, 'Bali': 0.05, 'Kalimantan Timur': 0.03, 'Riau': 0.02
    }

    KABUPATEN_MAP = {
        'Jawa Barat': ['Bandung', 'Bekasi', 'Bogor', 'Depok'],
        'Jawa Timur': ['Surabaya', 'Malang', 'Sidoarjo'],
        'Jawa Tengah': ['Semarang', 'Solo', 'Magelang'],
        'DKI Jakarta': ['Jakarta Pusat', 'Jakarta Selatan'],
        'Sumatera Utara': ['Medan', 'Deli Serdang'],
        'Banten': ['Tangerang', 'Serang'],
        'Sulawesi Selatan': ['Makassar', 'Gowa'],
        'Bali': ['Denpasar', 'Badung'],
        'Kalimantan Timur': ['Samarinda', 'Balikpapan'],
        'Riau': ['Pekanbaru', 'Kampar']
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.pathway_manager = ClinicalPathwayManager()
        self.drug_manager = DrugDispenseManager()
        self.history_manager = HistoricalClaimsManager()
        self.red_flags_calc = RedFlagsCalculator()
        self.graph_injector = GraphFraudInjector()

    def generate_nik_hash(self, participant_id: str) -> str:
        """Generate privacy-compliant hashed NIK."""
        base = f"NIK_{participant_id}_{self.seed}"
        return hashlib.sha256(base.encode()).hexdigest()[:16].upper()

    def assign_inacbg_tariff(self, row: pd.Series) -> Dict:
        """Assign INA-CBG code and tariff."""
        pathway = self.pathway_manager.get_pathway(row['kode_icd10'])

        if row['jenis_pelayanan'] == 'Rawat Inap':
            if row['room_class'] == 'Kelas III':
                tarif = pathway['tarif_kelas_3']
            elif row['room_class'] == 'Kelas II':
                tarif = pathway['tarif_kelas_2']
            else:
                tarif = pathway['tarif_kelas_1']
        else:
            tarif = pathway['tarif_kelas_3']

        return {
            'kode_tarif_inacbg': pathway['inacbg_code'],
            'tarif_inacbg': int(tarif),
            'clinical_pathway_name': pathway['name']
        }

    def generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        df = df.sort_values(['participant_id', 'tgl_pelayanan']).copy()

        df['time_diff_prev_claim'] = df.groupby('participant_id')['tgl_pelayanan'].diff().dt.days
        df['time_diff_prev_claim'] = df['time_diff_prev_claim'].fillna(999)

        df['claim_month'] = pd.to_datetime(df['tgl_pelayanan']).dt.month
        df['claim_quarter'] = pd.to_datetime(df['tgl_pelayanan']).dt.quarter

        df['rolling_avg_cost_30d'] = df.groupby('participant_id')['billed_amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        df['provider_monthly_claims'] = df.groupby(['dpjp_id', 'claim_month'])['claim_id'].transform('count')
        df['nik_hash_reuse_count'] = df.groupby('nik_hash')['nik_hash'].transform('count')

        return df

    def assemble_enhanced_claims(self, n_rows: int, providers_df: pd.DataFrame,
                                 participants_df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Generate enhanced claims."""
        logger.info(f"Generating {n_rows:,} enhanced claims...")

        provinces = list(self.PROVINCE_DISTRIBUTION.keys())
        province_probs = list(self.PROVINCE_DISTRIBUTION.values())
        selected_provinces = np.random.choice(provinces, size=n_rows, p=province_probs)

        claims = []
        for i in range(n_rows):
            prov = selected_provinces[i]
            kab_list = self.KABUPATEN_MAP.get(prov, ['Unknown'])

            prov_providers = providers_df[providers_df['provinsi'] == prov]
            if len(prov_providers) == 0:
                prov_providers = providers_df.sample(1)

            provider = prov_providers.sample(1).iloc[0]
            participant = participants_df.sample(1).iloc[0]

            icd10 = np.random.choice(list(self.pathway_manager.PATHWAYS.keys()))
            pathway = self.pathway_manager.get_pathway(icd10)

            jenis_pelayanan = pathway['service_type']
            lama_dirawat = np.random.poisson(pathway['typical_los']) if pathway['typical_los'] > 0 else 0

            cost_min, cost_max = pathway['cost_range']
            billed = int(np.random.uniform(cost_min, cost_max))

            claims.append({
                'claim_id': f'C{i+1:08d}',
                'episode_id': f'E{i+1:08d}',
                'participant_id': participant['participant_id'],
                'nik_hash': self.generate_nik_hash(participant['participant_id']),
                'age': participant['age'],
                'sex': participant['sex'],
                'faskes_id': provider['faskes_id'],
                'dpjp_id': provider['dpjp_id'],
                'faskes_level': provider['faskes_level'],
                'provinsi': prov,
                'kabupaten': np.random.choice(kab_list),
                'tgl_pelayanan': self.sample_date(year),
                'kode_icd10': icd10,
                'kode_prosedur': np.random.choice(pathway['typical_procedures']),
                'jenis_pelayanan': jenis_pelayanan,
                'room_class': self.assign_room_class(jenis_pelayanan),
                'lama_dirawat': lama_dirawat,
                'billed_amount': billed,
                'paid_amount': int(billed * np.random.uniform(0.7, 1.0)),
                'drug_cost': int(billed * np.random.beta(2, 8)),
                'procedure_cost': int(billed * np.random.beta(3, 5) * 0.6),
                'visit_count_30d': np.random.negative_binomial(2.5, 0.45),
                'kapitasi_flag': provider['faskes_level'] == 'FKTP' and np.random.rand() < 0.4,
                'referral_flag': np.random.rand() < 0.2,
                'referral_to_same_facility': False,
                'referral_to': '',
                'graph_pattern_id': '',
                'fraud_flag': 0,
                'fraud_type': 'none',
                'severity': 'none',
                'evidence_type': 'none'
            })

        df = pd.DataFrame(claims)

        # Assign INA-CBG
        inacbg_data = df.apply(self.assign_inacbg_tariff, axis=1, result_type='expand')
        df = pd.concat([df, inacbg_data], axis=1)

        # Add mapping columns
        df['diagnosis_name'] = df['kode_icd10'].map(ICD10_TO_NAME)
        df['procedure_name'] = df['kode_prosedur'].map(PROCEDURE_TO_NAME)
        df['inacbg_code'] = df['kode_tarif_inacbg']
        df['inacbg_tarif'] = df['tarif_inacbg']

        df['selisih_klaim'] = df['billed_amount'] - df['paid_amount']

        # Ensure costs don't exceed billed amount
        df['paid_amount'] = np.minimum(df['paid_amount'], df['billed_amount'])
        df['drug_cost'] = np.minimum(df['drug_cost'], df['billed_amount'])
        df['procedure_cost'] = np.minimum(df['procedure_cost'], df['billed_amount'] - df['drug_cost'])

        # Generate drug dispensing (MODUL B)
        df['obat_keluar'] = df.apply(
            lambda row: self.drug_manager.generate_drug_list(
                row['kode_icd10'],
                row['fraud_flag'] == 1,
                row['fraud_type']
            ), axis=1
        )

        df['obat_match_score'] = df.apply(
            lambda row: self.drug_manager.calculate_drug_match_score(
                row['obat_keluar'],
                row['kode_icd10']
            ), axis=1
        )

        # Generate temporal features
        df = self.generate_temporal_features(df)

        # Calculate enhanced deviation score (MODUL C)
        df['clinical_pathway_deviation_score'] = df.apply(
            self.pathway_manager.calculate_deviation_score_enhanced, axis=1
        )

        df['claim_ratio'] = df['billed_amount'] / (df['paid_amount'] + 1)

        logger.info(f"✅ Generated {len(df):,} enhanced claims")
        return df

    def sample_date(self, year: int) -> datetime:
        """Sample random date within year."""
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        days = (end - start).days
        return start + timedelta(days=np.random.randint(0, days + 1))

    def assign_room_class(self, jenis_pelayanan: str) -> str:
        """Assign room class for inpatient."""
        if jenis_pelayanan == 'Rawat Inap':
            return np.random.choice(['Kelas III', 'Kelas II', 'Kelas I', 'VIP'],
                                   p=[0.50, 0.30, 0.15, 0.05])
        return ''

    def make_providers(self, n_providers: int = 500) -> pd.DataFrame:
        """Generate provider pool."""
        providers = []
        provinces = list(self.PROVINCE_DISTRIBUTION.keys())
        province_probs = list(self.PROVINCE_DISTRIBUTION.values())

        for i in range(n_providers):
            faskes_level = np.random.choice(['FKTP', 'FKRTL'], p=[0.7, 0.3])
            prov = np.random.choice(provinces, p=province_probs)
            kab_list = self.KABUPATEN_MAP.get(prov, ['Unknown'])

            providers.append({
                'dpjp_id': f'D{i+1:05d}',
                'faskes_id': f'F{i+1:05d}',
                'faskes_level': faskes_level,
                'provinsi': prov,
                'kabupaten': np.random.choice(kab_list),
                'kapitasi_rate': int(np.random.gamma(2, 25000)) if faskes_level == 'FKTP' else 0
            })

        return pd.DataFrame(providers)

    def make_participants(self, n_participants: int = 50000) -> pd.DataFrame:
        """Generate participant pool."""
        ages = np.concatenate([
            np.random.randint(0, 20, size=int(n_participants * 0.3)),
            np.random.randint(20, 60, size=int(n_participants * 0.5)),
            np.random.randint(60, 90, size=int(n_participants * 0.2))
        ])
        np.random.shuffle(ages)

        return pd.DataFrame({
            'participant_id': [f'P{i+1:08d}' for i in range(n_participants)],
            'age': ages[:n_participants],
            'sex': np.random.choice(['M', 'F'], size=n_participants)
        })

    def inject_phantom_billing_enhanced(self, df: pd.DataFrame, n_phantom: int) -> pd.DataFrame:
        """
        MODUL D: Enhanced phantom billing injection with realistic patterns.

        Pattern 1: Spike after Normal (70%)
        - Patient with 5 normal outpatient claims
        - Suddenly has expensive inpatient claim with LOS=0

        Pattern 2: Drug Mismatch (30%)
        - Wrong drugs dispensed for diagnosis
        """
        logger.info(f"Injecting {n_phantom} enhanced phantom billing cases...")

        # Find eligible patients with sufficient history
        df_sorted = df.sort_values(['participant_id', 'tgl_pelayanan']).reset_index(drop=True)

        eligible_indices = []
        for participant_id, group in df_sorted.groupby('participant_id'):
            if len(group) >= 6:  # Need at least 6 claims
                # Check if first 5 are "normal"
                first_5 = group.iloc[:5]
                if (first_5['jenis_pelayanan'] == 'Rawat Jalan').all() and \
                   (first_5['rerata_billed_5x'].fillna(0) < 500_000).any():
                    # 6th claim can be phantom
                    eligible_indices.append(group.index[5])

        # Select random eligible indices
        n_pattern1 = int(n_phantom * 0.7)
        n_pattern2 = n_phantom - n_pattern1

        if len(eligible_indices) >= n_pattern1:
            pattern1_idx = np.random.choice(eligible_indices, size=n_pattern1, replace=False)
        else:
            pattern1_idx = []

        # Pattern 1: Spike after Normal
        for idx in pattern1_idx:
            rerata_prev = df.loc[idx, 'rerata_billed_5x'] if pd.notna(df.loc[idx, 'rerata_billed_5x']) else 300_000

            df.loc[idx, 'jenis_pelayanan'] = 'Rawat Inap'
            df.loc[idx, 'lama_dirawat'] = 0
            df.loc[idx, 'billed_amount'] = int(rerata_prev * np.random.uniform(3.0, 6.0))
            df.loc[idx, 'visit_count_30d'] = 0
            df.loc[idx, 'drug_cost'] = int(df.loc[idx, 'billed_amount'] * 0.15)
            df.loc[idx, 'procedure_cost'] = int(df.loc[idx, 'billed_amount'] * 0.6)

            # Wrong drugs
            df.loc[idx, 'obat_keluar'] = ['ceftriaxone', 'heparin']
            df.loc[idx, 'obat_match_score'] = 0.0

            df.loc[idx, 'fraud_flag'] = 1
            df.loc[idx, 'fraud_type'] = 'phantom_billing'
            df.loc[idx, 'severity'] = 'berat'
            df.loc[idx, 'evidence_type'] = 'system_anom'

        # Pattern 2: Drug Mismatch
        available_idx = df[(df['fraud_flag'] == 0) & (df['jenis_pelayanan'] == 'Rawat Jalan')].index.tolist()

        if len(available_idx) >= n_pattern2:
            pattern2_idx = np.random.choice(available_idx, size=n_pattern2, replace=False)

            for idx in pattern2_idx:
                icd = df.at[idx, 'kode_icd10']

                # Severe drug mismatch
                if icd in PHANTOM_MISMATCH_DRUGS:
                    df.at[idx, 'obat_keluar'] = PHANTOM_MISMATCH_DRUGS[icd][:2]
                else:
                    df.at[idx, 'obat_keluar'] = ['insulin', 'furosemide']

                df.at[idx, 'obat_match_score'] = 0.0
                df.at[idx, 'drug_cost'] = int(df.at[idx, 'billed_amount'] * 0.85)
                df.at[idx, 'billed_amount'] = int(df.at[idx, 'billed_amount'] * np.random.uniform(2.5, 5.0))
                df.at[idx, 'visit_count_30d'] = 0

                df.at[idx, 'fraud_flag'] = 1
                df.at[idx, 'fraud_type'] = 'phantom_billing'
                df.at[idx, 'severity'] = 'berat'
                df.at[idx, 'evidence_type'] = 'audit'

        # Recalculate derived columns
        phantom_mask = (df['fraud_flag'] == 1) & (df['fraud_type'] == 'phantom_billing')
        df.loc[phantom_mask, 'paid_amount'] = np.minimum(
            df.loc[phantom_mask, 'paid_amount'],
            df.loc[phantom_mask, 'billed_amount']
        )
        df.loc[phantom_mask, 'selisih_klaim'] = (
            df.loc[phantom_mask, 'billed_amount'] - df.loc[phantom_mask, 'paid_amount']
        )
        df.loc[phantom_mask, 'clinical_pathway_deviation_score'] = df.loc[phantom_mask].apply(
            self.pathway_manager.calculate_deviation_score_enhanced, axis=1
        )

        logger.info(f"✅ Injected {n_phantom} phantom billing cases")
        logger.info(f"  - Pattern 1 (Spike after Normal): {len(pattern1_idx)}")
        logger.info(f"  - Pattern 2 (Drug Mismatch): {n_pattern2}")

        return df

    def inject_traditional_fraud(self, df: pd.DataFrame, fraud_ratio: float) -> pd.DataFrame:
        """Inject traditional fraud types (10 types excluding phantom billing)."""
        logger.info(f"Injecting traditional fraud (ratio={fraud_ratio:.2%})...")

        n_total = len(df)
        n_fraud = int(n_total * fraud_ratio)

        # Fraud mix (excluding phantom_billing which is handled separately)
        fraud_mix = {
            'upcoding_diagnosis': 0.18,
            'cloning_claim': 0.10,
            'inflated_bill': 0.15,
            'service_unbundling': 0.12,
            'self_referral': 0.12,
            'repeat_billing': 0.10,
            'prolonged_los': 0.10,
            'room_manipulation': 0.08,
            'unnecessary_services': 0.03,
            'fake_license': 0.02
        }

        fraud_indices = np.random.choice(df[df['fraud_flag'] == 0].index,
                                        size=min(n_fraud, (df['fraud_flag'] == 0).sum()),
                                        replace=False)
        type_counts = {k: int(n_fraud * v) for k, v in fraud_mix.items()}

        idx_pointer = 0

        # 1. UPCODING DIAGNOSIS
        n_upcode = type_counts['upcoding_diagnosis']
        if idx_pointer + n_upcode <= len(fraud_indices):
            upcode_idx = fraud_indices[idx_pointer:idx_pointer + n_upcode]
            df.loc[upcode_idx, 'billed_amount'] = (df.loc[upcode_idx, 'billed_amount'] *
                                                    np.random.uniform(1.5, 2.5, size=len(upcode_idx))).astype(int)
            df.loc[upcode_idx, 'kode_icd10'] = np.random.choice(['I10', 'E11.9', 'J18.9'], size=len(upcode_idx))
            df.loc[upcode_idx, 'fraud_flag'] = 1
            df.loc[upcode_idx, 'fraud_type'] = 'upcoding_diagnosis'
            df.loc[upcode_idx, 'severity'] = 'sedang'
            df.loc[upcode_idx, 'evidence_type'] = 'audit'
            idx_pointer += n_upcode

        # 2. CLONING CLAIM
        n_clone = type_counts['cloning_claim']
        if idx_pointer + n_clone <= len(fraud_indices):
            clone_idx = fraud_indices[idx_pointer:idx_pointer + n_clone]
            for idx in clone_idx:
                random_claim = df.sample(1).iloc[0]
                df.loc[idx, 'kode_icd10'] = random_claim['kode_icd10']
                df.loc[idx, 'kode_prosedur'] = random_claim['kode_prosedur']
            df.loc[clone_idx, 'fraud_flag'] = 1
            df.loc[clone_idx, 'fraud_type'] = 'cloning_claim'
            df.loc[clone_idx, 'severity'] = 'berat'
            df.loc[clone_idx, 'evidence_type'] = 'audit'
            idx_pointer += n_clone

        # 3. INFLATED BILL
        n_inflated = type_counts['inflated_bill']
        if idx_pointer + n_inflated <= len(fraud_indices):
            inflated_idx = fraud_indices[idx_pointer:idx_pointer + n_inflated]
            df.loc[inflated_idx, 'drug_cost'] = (df.loc[inflated_idx, 'drug_cost'] *
                                                  np.random.uniform(2.5, 6.0, size=len(inflated_idx))).astype(int)
            df.loc[inflated_idx, 'billed_amount'] = (df.loc[inflated_idx, 'drug_cost'] +
                                                      df.loc[inflated_idx, 'procedure_cost']).astype(int)
            df.loc[inflated_idx, 'fraud_flag'] = 1
            df.loc[inflated_idx, 'fraud_type'] = 'inflated_bill'
            df.loc[inflated_idx, 'severity'] = 'sedang'
            df.loc[inflated_idx, 'evidence_type'] = 'system_anom'
            idx_pointer += n_inflated

         # 5. SERVICE UNBUNDLING
        n_unbundle = type_counts['service_unbundling']
        unbundle_idx = fraud_indices[idx_pointer:idx_pointer + n_unbundle]
        df.loc[unbundle_idx, 'procedure_cost'] = (df.loc[unbundle_idx, 'procedure_cost'] *
                                                   np.random.uniform(2.0, 4.0, size=len(unbundle_idx))).astype(int)
        df.loc[unbundle_idx, 'billed_amount'] = (df.loc[unbundle_idx, 'drug_cost'] +
                                                  df.loc[unbundle_idx, 'procedure_cost']).astype(int)
        df.loc[unbundle_idx, 'fraud_flag'] = 1
        df.loc[unbundle_idx, 'fraud_type'] = 'service_unbundling'
        df.loc[unbundle_idx, 'severity'] = 'sedang'
        df.loc[unbundle_idx, 'evidence_type'] = 'system_anom'
        idx_pointer += n_unbundle

        # 6. SELF-REFERRAL
        n_self_ref = type_counts['self_referral']
        self_ref_idx = fraud_indices[idx_pointer:idx_pointer + n_self_ref]
        df.loc[self_ref_idx, 'referral_flag'] = True
        df.loc[self_ref_idx, 'referral_to_same_facility'] = True
        df.loc[self_ref_idx, 'billed_amount'] = (df.loc[self_ref_idx, 'billed_amount'] *
                                                  np.random.uniform(1.3, 2.0, size=len(self_ref_idx))).astype(int)
        df.loc[self_ref_idx, 'fraud_flag'] = 1
        df.loc[self_ref_idx, 'fraud_type'] = 'self_referral'
        df.loc[self_ref_idx, 'severity'] = 'sedang'
        df.loc[self_ref_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_self_ref

        # 7. REPEAT BILLING
        n_repeat = type_counts['repeat_billing']
        repeat_idx = fraud_indices[idx_pointer:idx_pointer + n_repeat]
        df.loc[repeat_idx, 'fraud_flag'] = 1
        df.loc[repeat_idx, 'fraud_type'] = 'repeat_billing'
        df.loc[repeat_idx, 'severity'] = 'berat'
        df.loc[repeat_idx, 'evidence_type'] = 'system_anom'
        idx_pointer += n_repeat

        # 8. PROLONGED LOS
        n_prolong = type_counts['prolonged_los']
        prolong_idx = fraud_indices[idx_pointer:idx_pointer + n_prolong]

        ranap_mask = df.loc[prolong_idx, 'jenis_pelayanan'] == 'Rawat Inap'
        ranap_prolong = prolong_idx[ranap_mask]
        rajal_prolong = prolong_idx[~ranap_mask]

        if len(ranap_prolong) > 0:
            df.loc[ranap_prolong, 'lama_dirawat'] = np.random.randint(15, 60, size=len(ranap_prolong))
            df.loc[ranap_prolong, 'billed_amount'] = (df.loc[ranap_prolong, 'billed_amount'] *
                                                       np.random.uniform(2.0, 4.0, size=len(ranap_prolong))).astype(int)
            df.loc[ranap_prolong, 'fraud_flag'] = 1
            df.loc[ranap_prolong, 'fraud_type'] = 'prolonged_los'
            df.loc[ranap_prolong, 'severity'] = 'sedang'
            df.loc[ranap_prolong, 'evidence_type'] = 'audit'

        if len(rajal_prolong) > 0:
            df.loc[rajal_prolong, 'billed_amount'] = (df.loc[rajal_prolong, 'billed_amount'] *
                                                       np.random.uniform(1.5, 2.5, size=len(rajal_prolong))).astype(int)
            df.loc[rajal_prolong, 'fraud_flag'] = 1
            df.loc[rajal_prolong, 'fraud_type'] = 'upcoding_diagnosis'
            df.loc[rajal_prolong, 'severity'] = 'sedang'
            df.loc[rajal_prolong, 'evidence_type'] = 'audit'

        idx_pointer += n_prolong

        # 9. ROOM MANIPULATION
        n_room = type_counts['room_manipulation']
        room_idx = fraud_indices[idx_pointer:idx_pointer + n_room]

        ranap_mask = df.loc[room_idx, 'jenis_pelayanan'] == 'Rawat Inap'
        ranap_room = room_idx[ranap_mask]

        if len(ranap_room) > 0:
            df.loc[ranap_room, 'room_class'] = np.random.choice(['Kelas I', 'VIP'], size=len(ranap_room))
            df.loc[ranap_room, 'billed_amount'] = (df.loc[ranap_room, 'billed_amount'] *
                                                    np.random.uniform(1.4, 2.2, size=len(ranap_room))).astype(int)
            df.loc[ranap_room, 'fraud_flag'] = 1
            df.loc[ranap_room, 'fraud_type'] = 'room_manipulation'
            df.loc[ranap_room, 'severity'] = 'sedang'
            df.loc[ranap_room, 'evidence_type'] = 'audit'

        rajal_room = room_idx[~ranap_mask]
        if len(rajal_room) > 0:
            df.loc[rajal_room, 'drug_cost'] = (df.loc[rajal_room, 'drug_cost'] *
                                                np.random.uniform(2, 5, size=len(rajal_room))).astype(int)
            df.loc[rajal_room, 'billed_amount'] = (df.loc[rajal_room, 'billed_amount'] *
                                                    np.random.uniform(1.5, 2.5, size=len(rajal_room))).astype(int)
            df.loc[rajal_room, 'fraud_flag'] = 1
            df.loc[rajal_room, 'fraud_type'] = 'inflated_bill'
            df.loc[rajal_room, 'severity'] = 'sedang'
            df.loc[rajal_room, 'evidence_type'] = 'system_anom'

        idx_pointer += n_room

        # 10. UNNECESSARY SERVICES
        n_unnecessary = type_counts['unnecessary_services']
        unnecessary_idx = fraud_indices[idx_pointer:idx_pointer + n_unnecessary]
        df.loc[unnecessary_idx, 'procedure_cost'] = (df.loc[unnecessary_idx, 'procedure_cost'] *
                                                      np.random.uniform(1.5, 2.5, size=len(unnecessary_idx))).astype(int)
        df.loc[unnecessary_idx, 'visit_count_30d'] = np.random.randint(5, 15, size=len(unnecessary_idx))
        df.loc[unnecessary_idx, 'billed_amount'] = (df.loc[unnecessary_idx, 'drug_cost'] +
                                                     df.loc[unnecessary_idx, 'procedure_cost']).astype(int)
        df.loc[unnecessary_idx, 'fraud_flag'] = 1
        df.loc[unnecessary_idx, 'fraud_type'] = 'unnecessary_services'
        df.loc[unnecessary_idx, 'severity'] = np.random.choice(['ringan', 'sedang'], size=len(unnecessary_idx))
        df.loc[unnecessary_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_unnecessary

        # 11. FAKE LICENSE
        n_fake = type_counts['fake_license']
        fake_idx = fraud_indices[idx_pointer:idx_pointer + n_fake]
        df.loc[fake_idx, 'billed_amount'] = (df.loc[fake_idx, 'billed_amount'] *
                                              np.random.uniform(1.2, 2.0, size=len(fake_idx))).astype(int)
        df.loc[fake_idx, 'fraud_flag'] = 1
        df.loc[fake_idx, 'fraud_type'] = 'fake_license'
        df.loc[fake_idx, 'severity'] = 'berat'
        df.loc[fake_idx, 'evidence_type'] = 'whistleblower'

        # Recalculate
        df['paid_amount'] = np.minimum(df['paid_amount'], df['billed_amount'])
        df['selisih_klaim'] = df['billed_amount'] - df['paid_amount']

        df.loc[df['fraud_flag'] == 1, 'clinical_pathway_deviation_score_'] = df.loc[df['fraud_flag'] == 1].apply(
            self.pathway_manager.calculate_deviation_score_enhanced, axis=1
        )

        fraud_summary = df[df['fraud_flag'] == 1]['fraud_type'].value_counts().sort_index()
        logger.info(f"✅ Traditional fraud injection complete: {df['fraud_flag'].sum():,} fraudulent claims")
        logger.info("Fraud distribution by type:")
        for fraud_type, count in fraud_summary.items():
            percentage = count / df['fraud_flag'].sum() * 100
            logger.info(f"  - {fraud_type}: {count} ({percentage:.1f}%)")

  

        logger.info(f"✅ Traditional fraud injection complete: {(df['fraud_flag'] == 1).sum():,} cases")
        return df

    def generate_complete_dataset(self, n_rows: int, fraud_ratio: float,
                                  phantom_ratio: float, graph_fraud_ratio: float,
                                  year: int) -> pd.DataFrame:
        """Generate complete enhanced dataset."""
        logger.info("="*80)
        logger.info("BPJS V3.2 DATASET GENERATION - PHANTOM BILLING FOCUS")
        logger.info("="*80)

        # Generate base data
        providers_df = self.make_providers(500)
        participants_df = self.make_participants(50000)
        df = self.assemble_enhanced_claims(n_rows, providers_df, participants_df, year)

        # MODUL A: Extract 5 previous claims history
        df = self.history_manager.extract_prev_claims_features(df)

        # Inject traditional fraud (excluding phantom billing)
        df = self.inject_traditional_fraud(df, fraud_ratio - phantom_ratio)

        # MODUL D: Inject enhanced phantom billing
        n_phantom = int(n_rows * phantom_ratio)
        df = self.inject_phantom_billing_enhanced(df, n_phantom)

        # Inject graph fraud
        df = self.graph_injector.inject_all_graph_patterns(df, graph_fraud_ratio)

        # MODUL E: Calculate red flags
        df = self.red_flags_calc.calculate_red_flags(df)

        # Add ML features
        df = self.featurize(df)

        # MODUL F: Diagnostics
        self.print_phantom_diagnostics(df)

        return df

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixed version from artifact"""
        logger.info("Computing engineered features...")
        
        # Basic ratios
        df['drug_ratio'] = df['drug_cost'] / (df['billed_amount'] + 1)
        df['procedure_ratio'] = df['procedure_cost'] / (df['billed_amount'] + 1)
        
        # Date handling
        df['tgl_pelayanan'] = pd.to_datetime(df['tgl_pelayanan'], errors='coerce')
        df['year_month'] = df['tgl_pelayanan'].dt.to_period('M').astype(str)
        
        # 🔧 FIXED: Provider claim share with safe division
        try:
            provider_monthly_counts = (
                df.groupby(['dpjp_id', 'year_month'])
                  .size()
                  .reset_index(name='provider_claims')
            )
            
            monthly_total = (
                df.groupby('year_month')
                  .size()
                  .reset_index(name='total_claims')
            )
            
            provider_stats = provider_monthly_counts.merge(
                monthly_total, on='year_month', how='left'
            )
            
            if not provider_stats.empty:
                provider_stats['provider_claim_share'] = np.where(
                    provider_stats['total_claims'] > 0,
                    provider_stats['provider_claims'] / provider_stats['total_claims'],
                    0.0
                )
                
                df = df.merge(
                    provider_stats[['dpjp_id', 'year_month', 'provider_claim_share']],
                    on=['dpjp_id', 'year_month'],
                    how='left'
                )
                
                df['provider_claim_share'] = df['provider_claim_share'].fillna(0)
            else:
                df['provider_claim_share'] = 0.0
                
        except Exception as e:
            logger.error(f"Provider stats failed: {e}")
            df['provider_claim_share'] = 0.0
        
        # Cleanup
        df.drop(columns=['year_month'], inplace=True, errors='ignore')
        
        # Graph features
        df['phantom_node_flag'] = (
            (df['visit_count_30d'] == 0) &
            (df['drug_cost'] > 0.9 * df['billed_amount'])
        ).astype(int)
        
        return df
            
       



    def print_phantom_diagnostics(self, df: pd.DataFrame):
        """MODUL F: Print phantom billing diagnostics."""
        logger.info("\n" + "="*80)
        logger.info("PHANTOM BILLING DIAGNOSTICS")
        logger.info("="*80)

        phantom_df = df[df['fraud_type'] == 'phantom_billing']

        logger.info(f"\n📊 Total Phantom Billing Cases: {len(phantom_df):,}")
        logger.info(f"  - With visit_count_30d == 0: {(phantom_df['visit_count_30d'] == 0).sum():,}")
        logger.info(f"  - With LOS == 0: {(phantom_df['lama_dirawat'] == 0).sum():,}")

        logger.info(f"\n💊 Drug Match Score Statistics:")
        logger.info(f"  - Phantom billing avg: {phantom_df['obat_match_score'].mean():.3f}")
        logger.info(f"  - Legitimate avg: {df[df['fraud_flag']==0]['obat_match_score'].mean():.3f}")

        logger.info(f"\n🚩 Red Flags Distribution:")
        logger.info(f"  - phantom_suspect_score avg: {phantom_df['phantom_suspect_score'].mean():.3f}")
        logger.info(f"  - visit_suspicious_flag: {phantom_df['visit_suspicious_flag'].sum():,}")
        logger.info(f"  - obat_mismatch_flag: {phantom_df['obat_mismatch_flag'].sum():,}")
        logger.info(f"  - billing_spike_flag: {phantom_df['billing_spike_flag'].sum():,}")

        logger.info(f"\n📋 Sample Phantom Billing Cases (Top 5):")
        sample_cols = ['claim_id', 'billed_amount', 'visit_count_30d', 'obat_match_score',
                      'phantom_suspect_score', 'lama_dirawat']
        logger.info("\n" + phantom_df[sample_cols].head().to_string())

        logger.info("\n" + "="*80)


def save_outputs(df: pd.DataFrame, out_dir: str, params: Dict):
    """Save dataset and metadata."""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'bpjs_v32_phantom_enhanced.csv')
    parquet_path = os.path.join(out_dir, 'bpjs_v32_phantom_enhanced.parquet')
    meta_path = os.path.join(out_dir, 'metadata_v32.json')

    logger.info(f"Saving CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)

    logger.info(f"Saving Parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False, compression='snappy')

    metadata = {
        "dataset_info": {
            "name": "BPJS Healthcare Claims - Phantom Billing Enhanced",
            "version": "3.2-FINAL-PHANTOM",
            "description": "Enhanced dataset with deep phantom billing focus + 5 previous claims history",
            "created_date": datetime.now().isoformat(),
            "features_added": [
                "5 previous claims history per patient",
                "Drug dispensing with match score",
                "Enhanced clinical deviation scoring",
                "Multi-pattern phantom billing injection",
                "Red flags aggregate features",
                "Graph readiness indicators"
            ],
            "license": "MIT"
        },
        "generation_parameters": params,
        "statistics": {
            'n_rows': len(df),
            'n_fraud': int(df['fraud_flag'].sum()),
            'n_phantom': int((df['fraud_type'] == 'phantom_billing').sum()),
            'fraud_ratio': float(df['fraud_flag'].mean()),
            'fraud_type_distribution': df[df['fraud_flag']==1]['fraud_type'].value_counts().to_dict()
        }
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✅ All files saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate BPJS V3.2 Enhanced Dataset')
    parser.add_argument('--n_rows', type=int, default=100_000)
    parser.add_argument('--fraud_ratio', type=float, default=0.05, help='Total fraud ratio')
    parser.add_argument('--phantom_ratio', type=float, default=0.02, help='Phantom billing ratio')
    parser.add_argument('--graph_fraud_ratio', type=float, default=0.02)
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='./output')

    args = parser.parse_args()

    generator = BPJSDataGeneratorV32(seed=args.seed)

    df = generator.generate_complete_dataset(
        n_rows=args.n_rows,
        fraud_ratio=args.fraud_ratio,
        phantom_ratio=args.phantom_ratio,
        graph_fraud_ratio=args.graph_fraud_ratio,
        year=args.year
    )

    save_outputs(df, args.out_dir, vars(args))

    logger.info("\n✅ DATASET GENERATION COMPLETE!")
    logger.info(f"📊 Total rows: {len(df):,}")
    logger.info(f"🚨 Fraud cases: {df['fraud_flag'].sum():,} ({df['fraud_flag'].mean():.2%})")
    logger.info(f"👻 Phantom billing: {(df['fraud_type']=='phantom_billing').sum():,}")


# ata