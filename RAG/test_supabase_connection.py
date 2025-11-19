# test_supabase_connection.py
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"URL: {url}")
print(f"KEY: {key[:10]}...")

client = create_client(url, key)

# Test query ke tabel
try:
    result = client.table("bpjs_fraud_regulations").select("id, content").limit(1).execute()
    print(f"✅ Connection OK")
    print(f"Table access: OK")
    print(f"Data: {result.data}")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Table access: FAILED")