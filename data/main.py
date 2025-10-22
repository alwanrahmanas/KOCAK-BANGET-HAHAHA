if __name__ == "__main__":
  # from generate_synthetic_bpjs import BPJSDataGenerator

  gen = BPJSDataGenerator(seed=123)

  prov = gen.make_providers(1000)
  peserta = gen.make_participants(1000)
  klaim = gen.assemble_claims(100000,prov, peserta, 2024)
  klaim_fraud = gen.inject_fraud(klaim, fraud_ratio=0.05)
  klaim_features = gen.featurize(klaim_fraud)

  print(klaim_feature)
