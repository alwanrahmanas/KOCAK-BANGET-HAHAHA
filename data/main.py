if __name__ == "__main__":
  from generate_synthetic_bpjs import BPJSDataGenerator

  gen = BPJSDataGenerator(seed=123)

  prov = gen.make_providers(100)
  peserta = gen.make_participants(1000)
  klaim = gen.assemble_claims(5000, prov, peserta, 2024)

  
