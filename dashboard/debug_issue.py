"""
Debug script to identify serialization issues
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

def check_dataframe_types(df: pd.DataFrame):
    """Analyze DataFrame column types"""
    print("\n" + "="*60)
    print("DATAFRAME TYPE ANALYSIS")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:")
    
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else None
        sample_type = type(sample).__name__
        
        # Check for problematic types
        is_problematic = False
        issue = ""
        
        if dtype == 'object':
            # Check what's inside
            if sample is not None:
                if isinstance(sample, (list, dict, np.ndarray)):
                    is_problematic = True
                    issue = f"Contains {type(sample).__name__}"
        
        if isinstance(sample, (np.integer, np.floating, np.bool_)):
            is_problematic = True
            issue = "NumPy type"
        
        status = "⚠️ " if is_problematic else "✓ "
        print(f"{status} {col:30s} | {str(dtype):15s} | Sample type: {sample_type:15s} {issue}")

def test_json_serialization(df: pd.DataFrame):
    """Test if DataFrame can be JSON serialized"""
    print("\n" + "="*60)
    print("JSON SERIALIZATION TEST")
    print("="*60)
    
    records = df.to_dict(orient='records')
    
    print(f"\nTesting {len(records)} records...")
    
    for i, record in enumerate(records[:3]):  # Test first 3
        print(f"\n[Record {i}]")
        
        for key, value in record.items():
            try:
                json.dumps({key: value})
                status = "✓"
            except (TypeError, ValueError) as e:
                status = f"❌ {e}"
            
            value_type = type(value).__name__
            value_str = str(value)[:50] if value is not None else "None"
            
            print(f"  {status:40s} {key:30s} | {value_type:15s} | {value_str}")

def fix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fixes to make DataFrame JSON-safe"""
    print("\n" + "="*60)
    print("APPLYING FIXES")
    print("="*60)
    
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        dtype = df_fixed[col].dtype
        
        # Convert numpy types
        if pd.api.types.is_integer_dtype(dtype):
            print(f"  Converting {col} to Python int")
            df_fixed[col] = df_fixed[col].astype('Int64').astype(object)
            df_fixed[col] = df_fixed[col].apply(lambda x: int(x) if pd.notna(x) else None)
        
        elif pd.api.types.is_float_dtype(dtype):
            print(f"  Converting {col} to Python float")
            df_fixed[col] = df_fixed[col].astype('Float64').astype(object)
            df_fixed[col] = df_fixed[col].apply(
                lambda x: float(x) if pd.notna(x) and not (np.isinf(x) or np.isnan(x)) else None
            )
        
        elif pd.api.types.is_bool_dtype(dtype):
            print(f"  Converting {col} to Python bool")
            df_fixed[col] = df_fixed[col].apply(lambda x: bool(x) if pd.notna(x) else None)
        
        elif dtype == 'object':
            # Check for complex objects
            sample = df_fixed[col].dropna().iloc[0] if not df_fixed[col].dropna().empty else None
            
            if isinstance(sample, (list, dict, np.ndarray)):
                print(f"  Converting {col} complex objects to string")
                df_fixed[col] = df_fixed[col].apply(lambda x: str(x) if x is not None else None)
            
            elif isinstance(sample, (np.integer, np.floating, np.bool_)):
                print(f"  Converting {col} NumPy objects to Python types")
                df_fixed[col] = df_fixed[col].apply(
                    lambda x: x.item() if isinstance(x, (np.generic,)) and pd.notna(x) else None
                )
    
    return df_fixed

def main():
    """Run diagnostic on sample data"""
    # Load or generate sample data
    print("="*60)
    print("SERIALIZATION DEBUGGER")
    print("="*60)
    
    csv_path = input("\nEnter path to CSV file to test: ").strip()
    
    if not csv_path:
        print("No file specified, generating sample data...")
        
        # Generate sample with various types
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3], dtype=np.int64),
            'float_col': np.array([1.5, 2.5, 3.5], dtype=np.float64),
            'bool_col': np.array([True, False, True], dtype=np.bool_),
            'str_col': ['a', 'b', 'c'],
            'list_col': [[1, 2], [3, 4], [5, 6]],
            'dict_col': [{'x': 1}, {'y': 2}, {'z': 3}],
            'mixed_col': [1, 'text', None]
        })
    else:
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {csv_path}")
        except Exception as e:
            print(f"Failed to load file: {e}")
            return
    
    # Run diagnostics
    check_dataframe_types(df)
    test_json_serialization(df)
    
    # Apply fixes
    df_fixed = fix_dataframe(df)
    
    # Test again
    print("\n" + "="*60)
    print("AFTER FIXES")
    print("="*60)
    
    check_dataframe_types(df_fixed)
    test_json_serialization(df_fixed)
    
    # Try final JSON dump
    print("\n" + "="*60)
    print("FINAL JSON TEST")
    print("="*60)
    
    try:
        records = df_fixed.to_dict(orient='records')
        json_str = json.dumps(records, indent=2)
        print(f"✅ Success! Generated {len(json_str)} bytes of JSON")
        print(f"\nSample:\n{json_str[:500]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    main()