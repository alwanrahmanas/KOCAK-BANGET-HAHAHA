from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import sys
from pathlib import Path
import os
import math
import numpy as np
from datetime import datetime
import logging

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.inference_engine import BPJSFraudInferenceEngine
from RAG.bpjs_fraud_rag_system import BPJSFraudRAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== SANITIZER FUNCTIONS ====================
def sanitize_value(v):
    """Recursively sanitize values for JSON serialization"""
    if v is None:
        return None
    
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    
    if hasattr(v, 'dtype'):
        if isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(v)
        if isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(v, np.bool_):
            return bool(v)
    
    if isinstance(v, np.ndarray):
        return [sanitize_value(x) for x in v.tolist()]
    
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    
    if isinstance(v, dict):
        return {str(k): sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    
    try:
        return str(v)
    except:
        return None

def sanitize_df_for_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts"""
    records = df.to_dict(orient="records")
    
    sanitized = []
    for rec in records:
        clean_rec = {}
        for key, value in rec.items():
            clean_rec[str(key)] = sanitize_value(value)
        sanitized.append(clean_rec)
    
    return sanitized

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="BPJS Fraud Detection API",
    description="Backend untuk inference + RAG explanation",
    version="1.0.0"
)

# Global variable for RAG system
rag_system = None

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and load models on startup"""
    global rag_system
    
    logger.info("Starting BPJS Fraud Detection API...")
    
    try:
        logger.info("Initializing RAG system...")
        rag_system = BPJSFraudRAGSystem()
        
        logger.info("Setting up RAG components...")
        rag_system.setup(rebuild_vectors=False)
        
        logger.info("Loading ML models...")
        binary_dir = os.environ.get(
            "BINARY_ARTIFACTS_DIR",
            r"C:\Users\US3R\OneDrive\Dokumen\Data_Science\Project\MLInference\artifacts\binary_ensemble_20251119_190009"
        )
        multiclass_dir = os.environ.get(
            "MULTICLASS_ARTIFACTS_DIR",
            r"C:\Users\US3R\OneDrive\Dokumen\Data_Science\Project\MLInference\artifacts\multiclass_ensemble_20251119_190010"
        )
        
        rag_system.inference_engine.load_models(
            binary_artifacts_dir=binary_dir,
            multiclass_artifacts_dir=multiclass_dir
        )
        
        logger.info("✅ System ready for inference")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        logger.exception(e)
        rag_system = None

# ==================== HELPER FUNCTION ====================
def _generate_fraud_explanations(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Generate RAG explanations for fraud cases"""
    try:
        fraud_df = predictions_df[predictions_df['predicted_fraud'] == 1].copy()
        
        if len(fraud_df) == 0:
            logger.info("No fraud cases detected - skipping RAG generation")
            return pd.DataFrame()
        
        logger.info(f"Generating RAG explanations for {len(fraud_df)} fraud cases...")
        
        # ═══════════════════════════════════════════════════════════════
        # FIX: Pass ALL COLUMNS directly without subset/mapping
        # This keeps everything synced with the generator's expected keys!
        # ═══════════════════════════════════════════════════════════════
        fraud_cases = fraud_df.to_dict('records')  # Use all columns as-is
        
        # DIAGNOSTIC (optional, but recommended for debug)
        logger.info(f"[Debug] Columns passed to RAG: {list(fraud_df.columns)}")
        logger.info(f"[Debug] First fraud case: {fraud_cases[0] if fraud_cases else 'NO FRAUD CASE'}")
        
        explanations_df = rag_system.explain_fraud_cases_batch(fraud_cases, fraud_only=False)
        
        logger.info(f"✅ Generated {len(explanations_df)} RAG explanations")
        return explanations_df
        
    except Exception as e:
        logger.error(f"RAG explanation generation failed: {e}")
        logger.exception(e)
        return pd.DataFrame()


# ==================== ROUTES ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BPJS Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "API info (this page)",
            "/health": "Health check",
            "/predict": "POST - Fraud detection endpoint"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if rag_system is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "rag_ready": False,
                "inference_ready": False,
                "message": "System not initialized - check startup logs"
            }
        )
    
    return {
        "status": "running",
        "rag_ready": rag_system is not None,
        "inference_ready": rag_system.inference_engine.is_ready if rag_system else False,
        "rag_explainer_ready": rag_system.explanation_generator is not None if rag_system else False
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    enable_rag: bool = True
):
    """Predict fraud from uploaded CSV file"""
    
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Check /health endpoint."
        )
    
    if not rag_system.inference_engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. System unavailable."
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be CSV format"
        )
    
    try:
        logger.info(f"Processing file: {file.filename}")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
    
    try:
        logger.info("Running fraud detection inference...")
        result = rag_system.inference_engine.predict(
            df,
            generate_explanations=True,
            merge_with_input=True,
            evaluate=False
        )
        
        if isinstance(result, dict) and "predictions" in result:
            predictions_df = result["predictions"]
        else:
            predictions_df = result
        
        fraud_count = int((predictions_df['predicted_fraud'] == 1).sum())
        total_claims = int(len(predictions_df))
        
        logger.info(f"✅ Inference completed: {total_claims} predictions, {fraud_count} fraud cases")
        
        # RAG EXPLANATION
        audit_reports = None
        
        if enable_rag and fraud_count > 0:
            logger.info(f"Generating RAG audit reports for {fraud_count} fraud cases...")
            
            try:
                explanations_df = _generate_fraud_explanations(predictions_df)
                
                if not explanations_df.empty:
                    predictions_df = predictions_df.merge(
                        explanations_df[['claim_id', 'explanation_text', 'retrieved_docs']],
                        on='claim_id',
                        how='left'
                    )
                    
                    audit_reports = sanitize_df_for_json(explanations_df)
                    logger.info(f"✅ Generated {len(explanations_df)} RAG audit reports")
                    
            except Exception as e:
                logger.error(f"RAG explanation failed: {e}")
                logger.exception(e)
        
        safe_predictions = sanitize_df_for_json(predictions_df)
        
        response = {
            "success": True,
            "total_claims": total_claims,
            "fraud_detected": fraud_count,
            "predictions": safe_predictions
        }
        
        if audit_reports:
            response["audit_reports"] = audit_reports
            response["audit_reports_count"] = len(audit_reports)
        
        response = sanitize_value(response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    
    PORT = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )
