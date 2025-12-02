import os
from dotenv import load_dotenv
import streamlit as st

if os.path.exists(".env"):
    load_dotenv(".env")

def get_env(key, default=None):
    if "st" in globals() and hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)
