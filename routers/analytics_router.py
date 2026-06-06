import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import logging
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT_DIR / "uploads" / "charts"

# Status representation
class AnalyticsStatusResponse(BaseModel):
    status: str
    message: str
    last_trained_at: str
    charts_available: list

# -------------------------------------------------------------------------
# ENDPOINT: GET STATUS
# -------------------------------------------------------------------------
@router.get("/status", response_model=AnalyticsStatusResponse)
async def get_analytics_status():
    """
    Xem trạng thái của mô hình phân tích hành vi và danh sách biểu đồ đang có trên server.
    """
    charts = []
    if CHARTS_DIR.exists():
        charts = [f.name for f in CHARTS_DIR.iterdir() if f.is_file() and f.suffix == ".png"]
        
    # Get last modified time of one of the charts as proxy for last trained time
    last_trained = "Never"
    if charts:
        sample_chart = CHARTS_DIR / charts[0]
        mtime = sample_chart.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    return AnalyticsStatusResponse(
        status="active",
        message="Hệ thống phân tích hành vi và phân cụm người dùng sẵn sàng.",
        last_trained_at=last_trained,
        charts_available=charts
    )

# -------------------------------------------------------------------------
# ENDPOINT: TRIGGER RETRAIN
# -------------------------------------------------------------------------
def _run_retrain_task():
    try:
        from scripts.train_analytics import run_analytics_pipeline
        run_analytics_pipeline()
    except Exception as e:
        logger.error(f"Error in background analytics training: {e}")

@router.post("/train")
async def trigger_analytics_train(background_tasks: BackgroundTasks):
    """
    [ADMIN] Kích hoạt tiến trình huấn luyện mô hình phân cụm K-Means trực tuyến.
    Tiến trình sẽ kéo dữ liệu mới nhất từ Firestore, chạy ML phân khúc, cập nhật ngược
    nhãn phân khúc vào từng User Document trên Firestore và xuất các biểu đồ mới.
    """
    background_tasks.add_task(_run_retrain_task)
    return {
        "status": "success",
        "message": "Tiến trình huấn luyện mô hình phân cụm và kết xuất biểu đồ đã được kích hoạt chạy ngầm (background)."
    }

# -------------------------------------------------------------------------
# ENDPOINT: SERVE CHARTS
# -------------------------------------------------------------------------
@router.get("/charts/{chart_name}")
async def get_analytics_chart(chart_name: str):
    """
    Lấy trực tiếp file hình ảnh biểu đồ phân tích (demographics.png, spending_analysis.png, user_segmentation.png, model_evaluation.png).
    Có thể hiển thị trực tiếp lên trình duyệt hoặc ứng dụng di động Flutter qua URL!
    """
    # Ensure secure filename to prevent path traversal vulnerability
    clean_name = os.path.basename(chart_name)
    if not clean_name.endswith(".png"):
        clean_name += ".png"
        
    chart_path = CHARTS_DIR / clean_name
    
    if not chart_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Biểu đồ '{clean_name}' không tồn tại trên server. Vui lòng chạy API /train trước."
        )
        
    return FileResponse(
        path=str(chart_path),
        media_type="image/png",
        filename=clean_name
    )
