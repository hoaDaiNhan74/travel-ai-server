from fastapi import APIRouter, HTTPException
from models.schemas import PackingListRequest, StandardResponse

# Khai báo router
router = APIRouter()

@router.post("/packing-list", response_model=StandardResponse)
async def get_packing_list(request: PackingListRequest):
    try:
        # Import cục bộ để tránh circular import với main.py nếu ai_service được khởi tạo ở đó
        from main import ai_service
        
        result = await ai_service.generate_packing_list(request)
        return StandardResponse(
            status="success",
            message="Tạo danh sách hành lý thành công!",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo danh sách hành lý: {str(e)}")
