import os
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import logging
import traceback

# ─── Retrain Pipeline (Scheduler) ────────────────────────────────────────────
# Import lazy để tránh import TF 2 lần (TF đã được import ở trên)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.retrain_pipeline import start_background_scheduler, run_retrain_pipeline

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Import AI Services & Schemas ────────────────────────────────────────────
from services.ai_service import AiService
from models.schemas import TripPlanRequest, ChatRequest, StandardResponse

# ─── Import Routers ───────────────────────────────────────────────────────────
from routers import ai_router, trip_router

# ─── Global State ────────────────────────────────────────────────────────────
loaded_model = None
db = None
retrain_scheduler = None  # APScheduler BackgroundScheduler instance

# AiService được khởi tạo sau lifespan để nhận đúng model và db đã load xong
ai_service: AiService = AiService()


# ─── Pydantic Response Models (Recommendation API) ───────────────────────────
class RecommendationResponse(BaseModel):
    status: str
    user_id: str
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    firebase_connected: bool


# ─── Lifespan (Startup / Shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    1. Loads the TensorFlow Two-Tower Model ONCE.
    2. Initializes Firebase Admin SDK for Firestore access.
    3. Injects loaded model & db into AiService (Internal Wiring).
    4. Starts APScheduler — auto-retrain 02:00 AM every Sunday.
    """
    global loaded_model, db, retrain_scheduler

    # 1. Load TensorFlow Two-Tower Model
    model_path = "two_tower_model"
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model directory '{model_path}' not found!")
        else:
            logger.info(f"Loading TensorFlow model from '{model_path}'...")
            loaded_model = tf.saved_model.load(model_path)
            logger.info("\u2705 TensorFlow model loaded successfully.")
    except Exception as e:
        logger.error(f"\u274c Failed to load TF model: {e}")

    # 2. Initialize Firebase Admin SDK
    cred_path = "serviceAccountKey.json"
    try:
        if os.path.exists(cred_path):
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            db = firestore.client()
            logger.info("\u2705 Firebase Admin SDK initialized successfully.")
        else:
            logger.warning(f"'{cred_path}' not found \u2014 Firebase features disabled.")
    except Exception as e:
        logger.error(f"\u274c Firebase initialization failed: {e}")

    # 3. [INTERNAL WIRING] Inject TF model + Firestore DB v\u00e0o AiService
    ai_service.set_dependencies(loaded_model=loaded_model, db=db)
    logger.info("\U0001f517 Internal Wiring: AiService \u0111\u00e3 \u0111\u01b0\u1ee3c k\u1ebft n\u1ed1i v\u1edbi Recommendation Engine & Firestore.")

    # 4. [SCHEDULER] Kh\u1edfi \u0111\u1ed9ng APScheduler background
    # Sau m\u1ed7i l\u1ea7n retrain th\u00e0nh c\u00f4ng, _reload_model_after_retrain() s\u1ebd t\u1ef1 \u0111\u1ed9ng
    # n\u1ea1p l\u1ea1i model m\u1edbi v\u00e0o RAM m\u00e0 kh\u00f4ng c\u1ea7n restart server.
    try:
        retrain_scheduler = start_background_scheduler()
        logger.info("\u23f0 Retrain Scheduler kh\u1edfi \u0111\u1ed9ng (Chu Nhat 02:00 AM).")
    except Exception as e:
        logger.warning(f"\u26a0\ufe0f Kh\u00f4ng th\u1ec3 kh\u1edfi \u0111\u1ed9ng scheduler: {e}")

    yield

    # ─── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("\U0001f534 Server shutting down...")
    if retrain_scheduler and retrain_scheduler.running:
        retrain_scheduler.shutdown(wait=False)
        logger.info("Scheduler \u0111\u00e3 d\u1eebng.")


# ─── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="TripHero Unified API",
    version="2.0.0",
    description=(
        "Monolith API kết hợp Two-Tower ML Recommendation Engine "
        "và Groq AI (Llama-3) Trip Planner + Chatbot."
    ),
    lifespan=lifespan,
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ─────────────────────────────────────────────────────────
app.include_router(ai_router.router, prefix="/api/v1/ai", tags=["AI Extra Features"])
app.include_router(trip_router.router, prefix="/api/v1/trips", tags=["Trip Management"])


# ════════════════════════════════════════════════════════════════════════════════
# ─── SECTION 1: Root & Health Endpoints ───────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "TripHero Unified API is running 🚀",
        "version": "2.0.0",
        "services": {
            "recommendation_engine": loaded_model is not None,
            "firebase_connected": db is not None,
            "ai_planner": ai_service.client is not None,
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health-check endpoint — called by Flutter MLRecommendationService to warm up
    the Render.com free-tier server before the first recommendation request.
    """
    return HealthResponse(
        status="ok",
        model_loaded=loaded_model is not None,
        firebase_connected=db is not None,
    )


# ════════════════════════════════════════════════════════════════════════════════
# ─── SECTION 2: Recommendation Endpoints (Two-Tower ML Model) ─────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
async def get_recommendations(user_id: str):
    """
    Returns personalized destination recommendations for a given user_id
    using the pre-loaded Two-Tower TensorFlow model.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is not ready.")

    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    try:
        # 1. Convert user_id string → TF constant tensor
        input_tensor = tf.constant([user_id])

        # 2. Run inference (expects standard TFRS SavedModel export format)
        scores, destination_ids = loaded_model(input_tensor)

        # 3. Decode byte tensors → Python UTF-8 strings
        raw_ids = destination_ids[0].numpy()
        recommendations = [
            dest_id.decode("utf-8") if isinstance(dest_id, bytes) else str(dest_id)
            for dest_id in raw_ids
        ]

        logger.info(f"✅ Recommendations generated for user '{user_id}': {len(recommendations)} items")

        return RecommendationResponse(
            status="success",
            user_id=user_id,
            recommendations=recommendations[:10],  # Top 10
        )

    except Exception as e:
        logger.error(f"❌ Inference error for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation inference failed: {e}")


# ─── Backward-compatible alias ────────────────────────────────────────────────
@app.get(
    "/api/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations (Legacy)"],
    deprecated=True,
    summary="[Deprecated] Use /api/v1/recommend/{user_id} instead",
)
async def get_recommendations_legacy(user_id: str):
    """Legacy endpoint kept for backward compatibility with older Flutter clients."""
    return await get_recommendations(user_id)



# ════════════════════════════════════════════════════════════════════════════════
# ─── SECTION 4: AI Chatbot Endpoint ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/chat", response_model=StandardResponse, tags=["AI Chatbot"])
async def chat_with_ai(request: ChatRequest):
    """
    Chatbot du lịch AI dùng Groq (Llama-3).
    Hỗ trợ ngữ cảnh chuyến đi (tripContext) và lịch sử hội thoại (history).
    """
    logger.info(f"💬 Nhận tin nhắn chat: {request.message[:50]}...")
    try:
        reply = await ai_service.chat_with_ai(request)

        return StandardResponse(
            status="success",
            data={"reply": reply}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi chat: {e}")
        traceback.print_exc()
        return StandardResponse(
            status="error",
            message=f"Lỗi trợ lý ảo: {str(e)}",
            data=None
        )


# ════════════════════════════════════════════════════════════════════════════════
# ─── SECTION 5: Admin Endpoints (Retrain & Model Management) ──────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/admin/retrain", tags=["Admin"])
async def trigger_retrain_now():
    """
    [ADMIN] Kích hoạt retrain Two-Tower model ngay lập tức (không cần đợi lịch).
    Chạy bất đồng bộ trong background — không block request.

    WARNING: Chỉ dùng trong môi trường trusted. Cân nhắc thêm API key auth khi production.
    """
    import asyncio

    async def _run_retrain_async():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_retrain_pipeline)

    asyncio.create_task(_run_retrain_async())

    next_run = None
    if retrain_scheduler and retrain_scheduler.running:
        job = retrain_scheduler.get_job("two_tower_retrain")
        next_run = str(job.next_run_time) if job else None

    return {
        "status": "triggered",
        "message": "Retrain pipeline da duoc kich hoat chay trong background.",
        "next_scheduled_run": next_run,
    }


@app.get("/admin/scheduler/status", tags=["Admin"])
async def get_scheduler_status():
    """[ADMIN] Xem trạng thái của Retrain Scheduler."""
    if retrain_scheduler is None:
        return {"running": False, "message": "Scheduler chua duoc khoi dong."}

    job = retrain_scheduler.get_job("two_tower_retrain")
    return {
        "running":         retrain_scheduler.running,
        "job_id":          job.id if job else None,
        "next_run_time":   str(job.next_run_time) if job else None,
        "model_path":      "two_tower_model/",
        "model_loaded":    loaded_model is not None,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # reload=False is critical when serving large ML models.
    # reload=True would re-load the entire TF model on every file change → OOM crash.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
