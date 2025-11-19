"""
BPJS Fraud Detection Dataset Generator V3.1 - COMPLETE FIXED EDITION
=====================================================================
✅ FIXED: Probability sum error
✅ FIXED: Class integration issues
✅ COMPLETE: All 11 traditional fraud types + 5 graph fraud patterns
✅ COMPATIBLE: 100% ready for GraphXAIN inference

Version: 3.1-FINAL
Last Updated: 2024-11
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import json
import hashlib
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    def calculate_deviation_score(cls, row: pd.Series) -> float:
        """Calculate clinical pathway deviation score (0-1)."""
        pathway = cls.get_pathway(row['kode_icd10'])
        deviation_factors = []

        if pathway['typical_los'] > 0:
            los_deviation = abs(row['lama_dirawat'] - pathway['typical_los']) / pathway['typical_los']
            deviation_factors.append(min(los_deviation, 1.0))

        expected_cost = (pathway['cost_range'][0] + pathway['cost_range'][1]) / 2
        cost_deviation = abs(row['billed_amount'] - expected_cost) / expected_cost
        deviation_factors.append(min(cost_deviation, 1.0))

        if row['jenis_pelayanan'] != pathway['service_type']:
            deviation_factors.append(1.0)

        return np.mean(deviation_factors) if deviation_factors else 0.0


class FraudRuleEngine:
    """Fraud detection rules based on PerBPJS No. 6/2020."""

    FRAUD_RULES = {
        'upcoding_diagnosis': {
            'regulation': 'PerBPJS No. 6/2020 Pasal 12 ayat (2) huruf a',
            'description': 'Mengubah kode diagnosis untuk mendapatkan tarif lebih tinggi'
        },
        'phantom_billing': {
            'regulation': 'PerBPJS No. 6/2020 Pasal 12 ayat (2) huruf b',
            'description': 'Mengajukan klaim atas pelayanan yang tidak dilakukan'
        },
        'graph_ring_network': {
            'regulation': 'PerBPJS No. 6/2020 Pasal 13 (Kolusi)',
            'description': 'Jaringan rujukan melingkar (circular referral)'
        },
        'graph_star_center': {
            'regulation': 'PerBPJS No. 6/2020 Pasal 13 (Kolusi)',
            'description': 'Hub pusat dalam jaringan fraud'
        },
        'graph_chain_referral': {
            'regulation': 'PerBPJS No. 6/2020 Pasal 13 ayat (1)',
            'description': 'Rantai rujukan tidak wajar'
        }
    }

    @classmethod
    def get_rule(cls, fraud_type: str) -> Dict:
        return cls.FRAUD_RULES.get(fraud_type, {})


class GraphFraudInjector:
    """Inject graph-based fraud patterns for network analysis."""

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
        logger.info(f"Injecting {n_stars} star patterns (spokes={spokes_per_star})...")

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
            df.loc[hub_idx, 'evidence_type'] = 'graph_analysis'
            df.loc[hub_idx, 'graph_pattern_id'] = f'STAR_{star_id}'
            df.loc[hub_idx, 'billed_amount'] = int(df.loc[hub_idx, 'billed_amount'] * np.random.uniform(2.0, 4.0))

            for i, spoke_idx in enumerate(spoke_indices):
                spoke_provider = f'STAR{star_id}_S{i}'
                df.loc[spoke_idx, 'dpjp_id'] = spoke_provider
                df.loc[spoke_idx, 'referral_flag'] = True
                df.loc[spoke_idx, 'referral_to'] = hub_provider
                df.loc[spoke_idx, 'fraud_flag'] = 1
                df.loc[spoke_idx, 'fraud_type'] = 'graph_star_spoke'
                df.loc[spoke_idx, 'severity'] = 'sedang'
                df.loc[spoke_idx, 'evidence_type'] = 'graph_analysis'
                df.loc[spoke_idx, 'graph_pattern_id'] = f'STAR_{star_id}'
                df.loc[spoke_idx, 'billed_amount'] = int(df.loc[spoke_idx, 'billed_amount'] * np.random.uniform(1.3, 2.0))

        logger.info(f"✅ Injected {n_stars} star patterns")
        return df

    @staticmethod
    def inject_chain_referrals(df: pd.DataFrame, n_chains: int = 4, chain_length: int = 6) -> pd.DataFrame:
        """Inject linear referral chains."""
        logger.info(f"Injecting {n_chains} chain referrals (length={chain_length})...")

        available_idx = df[df['fraud_flag'] == 0].index.tolist()

        for chain_id in range(n_chains):
            if len(available_idx) < chain_length:
                break

            chain_indices = np.random.choice(available_idx, size=chain_length, replace=False)
            available_idx = [i for i in available_idx if i not in chain_indices]
            providers = [f'CHAIN{chain_id}_P{i}' for i in range(chain_length)]

            for i, idx in enumerate(chain_indices):
                current_provider = providers[i]
                next_provider = providers[i + 1] if i < chain_length - 1 else None

                df.loc[idx, 'dpjp_id'] = current_provider
                df.loc[idx, 'referral_flag'] = True if next_provider else False
                df.loc[idx, 'referral_to'] = next_provider if next_provider else ''
                df.loc[idx, 'fraud_flag'] = 1
                df.loc[idx, 'fraud_type'] = 'graph_chain_referral'
                df.loc[idx, 'severity'] = 'sedang'
                df.loc[idx, 'evidence_type'] = 'graph_analysis'
                df.loc[idx, 'graph_pattern_id'] = f'CHAIN_{chain_id}'
                multiplier = 1.2 + (i * 0.2)
                df.loc[idx, 'billed_amount'] = int(df.loc[idx, 'billed_amount'] * multiplier)

        logger.info(f"✅ Injected {n_chains} chain referrals")
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

    @staticmethod
    def inject_bipartite_scheme(df: pd.DataFrame, n_schemes: int = 3,
                                providers_per_scheme: int = 3, patients_per_scheme: int = 5) -> pd.DataFrame:
        """Inject bipartite fraud schemes (providers-patients collusion)."""
        logger.info(f"Injecting {n_schemes} bipartite schemes...")

        available_idx = df[df['fraud_flag'] == 0].index.tolist()

        for scheme_id in range(n_schemes):
            needed = providers_per_scheme * patients_per_scheme
            if len(available_idx) < needed:
                break

            scheme_indices = np.random.choice(available_idx, size=needed, replace=False)
            available_idx = [i for i in available_idx if i not in scheme_indices]

            providers = [f'BIPART{scheme_id}_PROV{i}' for i in range(providers_per_scheme)]
            patients = [f'BIPART{scheme_id}_PAT{i}' for i in range(patients_per_scheme)]

            idx_counter = 0
            for provider in providers:
                for patient in patients:
                    if idx_counter >= len(scheme_indices):
                        break

                    idx = scheme_indices[idx_counter]
                    df.loc[idx, 'dpjp_id'] = provider
                    df.loc[idx, 'participant_id'] = patient
                    df.loc[idx, 'fraud_flag'] = 1
                    df.loc[idx, 'fraud_type'] = 'graph_bipartite_collusion'
                    df.loc[idx, 'severity'] = 'berat'
                    df.loc[idx, 'evidence_type'] = 'graph_analysis'
                    df.loc[idx, 'graph_pattern_id'] = f'BIPART_{scheme_id}'
                    df.loc[idx, 'billed_amount'] = int(df.loc[idx, 'billed_amount'] * np.random.uniform(2.5, 5.0))
                    df.loc[idx, 'visit_count_30d'] = 0

                    idx_counter += 1

        logger.info(f"✅ Injected {n_schemes} bipartite schemes")
        return df

    @classmethod
    def inject_all_graph_patterns(cls, df: pd.DataFrame, graph_fraud_ratio: float = 0.02) -> pd.DataFrame:
        """Master function to inject all graph fraud patterns."""
        logger.info(f"=" * 80)
        logger.info("INJECTING GRAPH FRAUD PATTERNS")
        logger.info(f"=" * 80)

        # Ensure columns exist
        if 'referral_to' not in df.columns:
            df['referral_to'] = ''
        else:
            df['referral_to'] = df['referral_to'].fillna('')

        if 'graph_pattern_id' not in df.columns:
            df['graph_pattern_id'] = ''
        else:
            df['graph_pattern_id'] = df['graph_pattern_id'].fillna('')

        n_total = len(df)
        n_graph_fraud = int(n_total * graph_fraud_ratio)

        logger.info(f"Target graph fraud cases: {n_graph_fraud:,} ({graph_fraud_ratio:.2%})")

        n_rings = max(1, int(n_graph_fraud * 0.15 / 5))
        n_stars = max(1, int(n_graph_fraud * 0.30 / 9))
        n_chains = max(1, int(n_graph_fraud * 0.20 / 6))
        n_cliques = max(1, int(n_graph_fraud * 0.15 / 4))
        n_bipartite = max(1, int(n_graph_fraud * 0.20 / 15))

        df = cls.inject_ring_network(df, n_rings=n_rings, ring_size=5)
        df = cls.inject_star_pattern(df, n_stars=n_stars, spokes_per_star=8)
        df = cls.inject_chain_referrals(df, n_chains=n_chains, chain_length=6)
        df = cls.inject_clique_collusion(df, n_cliques=n_cliques, clique_size=4)
        df = cls.inject_bipartite_scheme(df, n_schemes=n_bipartite,
                                        providers_per_scheme=3, patients_per_scheme=5)

        graph_fraud_count = df[df['fraud_type'].str.contains('graph_', na=False)].shape[0]
        logger.info(f"=" * 80)
        logger.info(f"✅ Total graph fraud cases injected: {graph_fraud_count:,}")
        logger.info(f"=" * 80)

        return df


# ========== MAIN GENERATOR CLASS ==========

class BPJSDataGeneratorV3:
    """Enhanced BPJS dataset generator with graph fraud patterns."""

    # ✅ FIXED: Probabilities now sum to 1.0
    PROVINCE_DISTRIBUTION = {
        'Jawa Barat': 0.20,
        'Jawa Timur': 0.18,
        'Jawa Tengah': 0.15,
        'DKI Jakarta': 0.12,
        'Sumatera Utara': 0.10,
        'Banten': 0.08,
        'Sulawesi Selatan': 0.07,
        'Bali': 0.05,
        'Kalimantan Timur': 0.03,
        'Riau': 0.02
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
        self.rule_engine = FraudRuleEngine()
        self.graph_injector = GraphFraudInjector()

    def generate_nik_hash(self, participant_id: str) -> str:
        """Generate privacy-compliant hashed NIK."""
        base = f"NIK_{participant_id}_{self.seed}"
        return hashlib.sha256(base.encode()).hexdigest()[:16].upper()

    def assign_inacbg_tariff(self, row: pd.Series) -> Dict:
        """Assign INA-CBG code and tariff based on diagnosis."""
        pathway = self.pathway_manager.get_pathway(row['kode_icd10'])

        if row['jenis_pelayanan'] == 'Rawat Inap':
            if row['room_class'] == 'Kelas III':
                tarif = pathway['tarif_kelas_3']
            elif row['room_class'] == 'Kelas II':
                tarif = pathway['tarif_kelas_2']
            elif row['room_class'] in ['Kelas I', 'VIP']:
                tarif = pathway['tarif_kelas_1']
            else:
                tarif = pathway['tarif_kelas_3']
        else:
            tarif = pathway['tarif_kelas_3']

        return {
            'kode_tarif_inacbg': pathway['inacbg_code'],
            'tarif_inacbg': int(tarif),
            'clinical_pathway_name': pathway['name']
        }

    def generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced temporal features."""
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
        """Generate enhanced claims with all compliance features."""

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
                'referral_to': '',  # ✅ GraphXAIN compatible
                'graph_pattern_id': '',  # ✅ GraphXAIN compatible
                'fraud_flag': 0,
                'fraud_type': 'none',
                'severity': 'none',
                'evidence_type': 'none'
            })

        df = pd.DataFrame(claims)

        inacbg_data = df.apply(self.assign_inacbg_tariff, axis=1, result_type='expand')
        df = pd.concat([df, inacbg_data], axis=1)

        df['selisih_klaim'] = df['billed_amount'] - df['paid_amount']

        df['paid_amount'] = np.minimum(df['paid_amount'], df['billed_amount'])
        df['drug_cost'] = np.minimum(df['drug_cost'], df['billed_amount'])
        df['procedure_cost'] = np.minimum(df['procedure_cost'], df['billed_amount'] - df['drug_cost'])

        df = self.generate_temporal_features(df)

        df['clinical_pathway_deviation_score'] = df.apply(
            self.pathway_manager.calculate_deviation_score, axis=1
        )

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
        """Generate provider pool with provincial distribution."""
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

    def inject_fraud(self, df: pd.DataFrame, fraud_ratio: float = 0.03) -> pd.DataFrame:
        """Inject 11 types of traditional fraud."""
        logger.info(f"Injecting traditional fraud with ratio {fraud_ratio:.2%}...")

        n_total = len(df)
        n_fraud = int(n_total * fraud_ratio)

        fraud_mix = {
            'upcoding_diagnosis': 0.15,
            'phantom_billing': 0.12,
            'cloning_claim': 0.08,
            'inflated_bill': 0.12,
            'service_unbundling': 0.10,
            'self_referral': 0.10,
            'repeat_billing': 0.08,
            'prolonged_los': 0.10,
            'room_manipulation': 0.08,
            'unnecessary_services': 0.05,
            'fake_license': 0.02
        }

        fraud_indices = np.random.choice(n_total, size=n_fraud, replace=False)
        type_counts = {k: int(n_fraud * v) for k, v in fraud_mix.items()}

        idx_pointer = 0

        # 1. UPCODING DIAGNOSIS
        n_upcode = type_counts['upcoding_diagnosis']
        upcode_idx = fraud_indices[idx_pointer:idx_pointer + n_upcode]
        df.loc[upcode_idx, 'billed_amount'] = (df.loc[upcode_idx, 'billed_amount'] *
                                                np.random.uniform(1.5, 2.5, size=len(upcode_idx))).astype(int)
        df.loc[upcode_idx, 'kode_icd10'] = np.random.choice(['I10', 'E11.9', 'J18.9'], size=len(upcode_idx))
        df.loc[upcode_idx, 'fraud_flag'] = 1
        df.loc[upcode_idx, 'fraud_type'] = 'upcoding_diagnosis'
        df.loc[upcode_idx, 'severity'] = np.random.choice(['sedang', 'berat'], size=len(upcode_idx), p=[0.4, 0.6])
        df.loc[upcode_idx, 'evidence_type'] = 'audit'
        idx_pointer += n_upcode

        # 2. PHANTOM BILLING
        n_phantom = type_counts['phantom_billing']
        phantom_idx = fraud_indices[idx_pointer:idx_pointer + n_phantom]
        df.loc[phantom_idx, 'billed_amount'] = (df.loc[phantom_idx, 'billed_amount'] *
                                                 np.random.uniform(2.5, 6.0, size=len(phantom_idx))).astype(int)
        df.loc[phantom_idx, 'visit_count_30d'] = np.random.choice([0, 1], size=len(phantom_idx))
        df.loc[phantom_idx, 'fraud_flag'] = 1
        df.loc[phantom_idx, 'fraud_type'] = 'phantom_billing'
        df.loc[phantom_idx, 'severity'] = 'berat'
        df.loc[phantom_idx, 'evidence_type'] = np.random.choice(['system_anom', 'audit'], size=len(phantom_idx))
        idx_pointer += n_phantom

        # 3. CLONING CLAIM
        n_clone = type_counts['cloning_claim']
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

        # 4. INFLATED BILL
        n_inflated = type_counts['inflated_bill']
        inflated_idx = fraud_indices[idx_pointer:idx_pointer + n_inflated]
        df.loc[inflated_idx, 'drug_cost'] = (df.loc[inflated_idx, 'drug_cost'] *
                                              np.random.uniform(2.5, 8.0, size=len(inflated_idx))).astype(int)
        df.loc[inflated_idx, 'procedure_cost'] = (df.loc[inflated_idx, 'procedure_cost'] *
                                                   np.random.uniform(1.5, 3.0, size=len(inflated_idx))).astype(int)
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

        df.loc[df['fraud_flag'] == 1, 'clinical_pathway_deviation_score'] = df.loc[df['fraud_flag'] == 1].apply(
            self.pathway_manager.calculate_deviation_score, axis=1
        )

        fraud_summary = df[df['fraud_flag'] == 1]['fraud_type'].value_counts().sort_index()
        logger.info(f"✅ Traditional fraud injection complete: {df['fraud_flag'].sum():,} fraudulent claims")
        logger.info("Fraud distribution by type:")
        for fraud_type, count in fraud_summary.items():
            percentage = count / df['fraud_flag'].sum() * 100
            logger.info(f"  - {fraud_type}: {count} ({percentage:.1f}%)")

        return df

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for ML."""
        logger.info("Computing engineered features...")

        df['claim_ratio'] = df['billed_amount'] / (df['paid_amount'] + 1)
        df['drug_ratio'] = df['drug_cost'] / (df['billed_amount'] + 1)
        df['procedure_ratio'] = df['procedure_cost'] / (df['billed_amount'] + 1)

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

        logger.info("✅ Feature engineering complete")

        return df

    def generate_complete_dataset(self, n_rows: int, fraud_ratio: float,
                                  graph_fraud_ratio: float, year: int) -> pd.DataFrame:
        """Generate complete dataset with traditional fraud + graph fraud."""
        logger.info("Generating providers and participants...")
        providers_df = self.make_providers(500)
        participants_df = self.make_participants(50000)

        logger.info(f"Assembling {n_rows:,} base claims...")
        df = self.assemble_enhanced_claims(n_rows, providers_df, participants_df, year)

        # Inject traditional fraud (11 types)
        df = self.inject_fraud(df, fraud_ratio=fraud_ratio)

        # Inject graph fraud patterns (5 types)
        df = self.graph_injector.inject_all_graph_patterns(df, graph_fraud_ratio=graph_fraud_ratio)

        # Add ML features
        df = self.featurize(df)

        return df


# ========== UTILITY FUNCTIONS ==========


def print_summary(df: pd.DataFrame):
    """Print dataset summary."""
    print("\n" + "=" * 80)
    print("BPJS V3.0-COMPLETE DATASET SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total Claims: {len(df):,}")
    print(f"🚨 Fraudulent Claims: {df['fraud_flag'].sum():,} ({df['fraud_flag'].mean():.2%})")
    print(f"✅ Legitimate Claims: {(df['fraud_flag'] == 0).sum():,} ({(df['fraud_flag'] == 0).mean():.2%})")

    print("\n🔍 Fraud Type Distribution:")
    fraud_dist = df[df['fraud_flag'] == 1]['fraud_type'].value_counts().sort_values(ascending=False)
    for fraud_type, count in fraud_dist.items():
        pct = count / df['fraud_flag'].sum() * 100
        logger.info(f"  {fraud_type:25s}: {count:5d} ({pct:5.1f}%)")

    print("\n💰 Monetary Statistics (IDR):")
    print(df[['billed_amount', 'paid_amount', 'drug_cost', 'procedure_cost']].describe().round(0))

    print("\n🏥 Service Type Distribution:")
    print(df['jenis_pelayanan'].value_counts())

    print("\n🎯 Clinical Pathway Deviation (Fraud vs Legitimate):")
    print(f"  Fraud avg deviation: {df[df['fraud_flag']==1]['clinical_pathway_deviation_score'].mean():.3f}")
    print(f"  Legit avg deviation: {df[df['fraud_flag']==0]['clinical_pathway_deviation_score'].mean():.3f}")

    print("\n" + "=" * 80)

def print_summary_with_graph(df: pd.DataFrame):
    """Print dataset summary including graph fraud."""
    print("\n" + "=" * 80)
    print("BPJS V3.1-GRAPH DATASET SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total Claims: {len(df):,}")
    print(f"🚨 Total Fraudulent Claims: {df['fraud_flag'].sum():,} ({df['fraud_flag'].mean():.2%})")

    # Traditional fraud
    trad_fraud = df[~df['fraud_type'].str.contains('graph_', na=False) & (df['fraud_flag'] == 1)]
    print(f"  ├─ Traditional Fraud: {len(trad_fraud):,}")

    # Graph fraud
    graph_fraud = df[df['fraud_type'].str.contains('graph_', na=False)]
    print(f"  └─ Graph Fraud: {len(graph_fraud):,}")

    print("\n🕸️  Graph Fraud Type Distribution:")
    if len(graph_fraud) > 0:
        graph_dist = graph_fraud['fraud_type'].value_counts()
        for fraud_type, count in graph_dist.items():
            pct = count / len(graph_fraud) * 100
            print(f"  {fraud_type:30s}: {count:5d} ({pct:5.1f}%)")

    print("\n🔍 Traditional Fraud Type Distribution:")
    if len(trad_fraud) > 0:
        trad_dist = trad_fraud['fraud_type'].value_counts().head(10)
        for fraud_type, count in trad_dist.items():
            pct = count / len(trad_fraud) * 100
            print(f"  {fraud_type:30s}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)


def save_outputs(df: pd.DataFrame, out_dir: str, params: Dict):
    """Save dataset and metadata."""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'bpjs_v3_with_graph_fraud.csv')
    parquet_path = os.path.join(out_dir, 'bpjs_v3_with_graph_fraud.parquet')
    meta_path = os.path.join(out_dir, 'metadata_v3_graph.json')

    logger.info(f"Saving CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)

    logger.info(f"Saving Parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False, compression='snappy')

    metadata = {
        "dataset_info": {
            "name": "BPJS Healthcare Claims Fraud Detection Dataset",
            "version": "3.1-GRAPH-COMPLETE",
            "description": "Complete synthetic dataset with 11 traditional + 5 graph fraud patterns",
            "created_date": datetime.now().isoformat(),
            "license": "MIT"
        },
        "generation_parameters": params,
        "statistics": {
            'n_rows': len(df),
            'n_fraud': int(df['fraud_flag'].sum()),
            'fraud_ratio': float(df['fraud_flag'].mean()),
            'fraud_type_distribution': df[df['fraud_flag']==1]['fraud_type'].value_counts().to_dict()
        }
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✅ All files saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate BPJS V3.1 Dataset with Graph Fraud')
    parser.add_argument('--n_rows', type=int, default=100_000)
    parser.add_argument('--fraud_ratio', type=float, default=0.03, help='Traditional fraud ratio')
    parser.add_argument('--graph_fraud_ratio', type=float, default=0.02, help='Graph fraud ratio')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='./output')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BPJS V3.1-GRAPH DATASET GENERATOR")
    logger.info("=" * 80)

    generator = BPJSDataGeneratorV3(seed=args.seed)

    df = generator.generate_complete_dataset(
        n_rows=args.n_rows,
        fraud_ratio=args.fraud_ratio,
        graph_fraud_ratio=args.graph_fraud_ratio,
        year=args.year
    )

    # ✅ FINAL VALIDATION: Ensure GraphXAIN compatibility
    required_cols = ['referral_to', 'graph_pattern_id']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')

    logger.info("\n✅ GRAPHXAIN COMPATIBILITY CHECK:")
    logger.info(f"  - 'referral_to' column: {'✓' if 'referral_to' in df.columns else '✗'}")
    logger.info(f"  - 'graph_pattern_id' column: {'✓' if 'graph_pattern_id' in df.columns else '✗'}")
    logger.info(f"  - Null values in referral_to: {df['referral_to'].isna().sum()}")
    logger.info(f"  - Null values in graph_pattern_id: {df['graph_pattern_id'].isna().sum()}")

    # Save
    save_outputs(df, args.out_dir, vars(args))

    # Summary
    print_summary_with_graph(df)

    logger.info("=" * 80)
    logger.info("✅ DATASET GENERATION COMPLETE!")
    logger.info("=" * 80)


# if __name__ == '__main__':
#     main()