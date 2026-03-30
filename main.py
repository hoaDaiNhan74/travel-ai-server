import os
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model
loaded_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the TensorFlow model once to optimize RAM and inference time.
    """
    global loaded_model
    model_path = "two_tower_model"
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model directory '{model_path}' not found!")
            raise FileNotFoundError(f"Model directory '{model_path}' not found.")
            
        logger.info(f"Loading TensorFlow model from {model_path}...")
        loaded_model = tf.saved_model.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # In a real production app, we might want to shut down if the model fails to load
        
    yield
    # Clean up (if needed) on shutdown
    logger.info("Shutting down server...")

# Initialize FastAPI with lifespan
app = FastAPI(title="TripHero Recommendation API", lifespan=lifespan)

# Initialize Firebase Admin SDK
try:
    cred_path = "serviceAccountKey.json"
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully.")
    else:
        logger.warning(f"'{cred_path}' not found. Firebase features will be disabled.")
except Exception as e:
    logger.error(f"Firebase initialization failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "TripHero Recommendation API is running", "model_loaded": loaded_model is not None}

@app.get("/api/recommend/{user_id}")
async def get_recommendations(user_id: str):
    """
    Predict recommendations for a given user_id using the Two-Tower model.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    try:
        # 1. Convert user_id (string) to tensor
        # TFRS models usually expect a 1D tensor of IDs
        input_tensor = tf.constant([user_id])
        
        # 2. Perform Inference
        # The model usually returns (scores, destination_ids)
        # Note: Depending on how the model was exported, you might need to call specific signatures
        # or handle the output dictionary. Assuming standard TFRS export.
        scores, destination_ids = loaded_model(input_tensor)
        
        # 3. Process Tensor output
        # Convert Byte Tensors to Python Strings (UTF-8)
        # destination_ids is likely a 2D tensor (batch_size, top_k)
        recommendations = []
        
        # Extract the results from the tensor
        # We take the first element since batch size is 1
        raw_ids = destination_ids[0].numpy()
        
        for dest_id in raw_ids:
            if isinstance(dest_id, bytes):
                recommendations.append(dest_id.decode('utf-8'))
            else:
                recommendations.append(str(dest_id))
        
        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations[:10]  # Return top 10 as per typical retrieval
        }
        
    except Exception as e:
        logger.error(f"Inference error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # In Windows, use reload=False or wrap in main check for large ML models
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
