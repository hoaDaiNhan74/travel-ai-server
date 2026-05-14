from fastapi import APIRouter, HTTPException, Request as FastApiRequest
from models.schemas import TripPlanRequest, PackingListRequest, StandardResponse, JoinTripRequest
from typing import Dict, Any
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/{trip_id}/packing-list", response_model=StandardResponse)
async def generate_and_save_packing_list(trip_id: str, request: PackingListRequest):
    """
    Lazy Loading API: Sinh hành lý và cập nhật vào chuyến đi đã tồn tại.
    """
    try:
        from main import ai_service
        
        # 1. Kiểm tra sự tồn tại của trip trên Firestore
        if not ai_service._db:
             raise HTTPException(status_code=500, detail="Firestore không khả dụng")
             
        trip_ref = ai_service._db.collection("trips").document(trip_id)
        trip_doc = trip_ref.get()
        
        if not trip_doc.exists:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy chuyến đi với ID: {trip_id}")

        # 2. Gọi AI sinh danh sách hành lý
        result = await ai_service.generate_packing_list(request)
        ai_categories = result.get("categories", [])
        
        # 3. Chuẩn hóa sang định dạng checklist (is_checked: false)
        formatted_packing_list = []
        for cat in ai_categories:
            items = cat.get("items", [])
            formatted_items = []
            for item in items:
                if isinstance(item, str):
                    formatted_items.append({"name": item, "is_checked": False})
                else:
                    formatted_items.append(item)
            
            formatted_packing_list.append({
                "name": cat.get("name", ""),
                "items": formatted_items
            })

        # 4. Cập nhật trực tiếp vào Firestore
        trip_ref.update({
            "packing_list": formatted_packing_list
        })
        
        logger.info(f"✅ Đã cập nhật packing_list cho Trip: {trip_id}")

        return StandardResponse(
            status="success",
            message="Đã tạo và cập nhật danh sách hành lý!",
            data={"packing_list": formatted_packing_list}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Lỗi khi sinh hành lý cho Trip {trip_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi hệ thống khi tạo hành lý: {str(e)}"
        )

@router.post("/generate", response_model=StandardResponse, tags=["AI Trip Planner"])
async def generate_trip(request: TripPlanRequest):
    """
    Tạo lịch trình du lịch AI cá nhân hóa dùng Groq (Llama-3).
    """
    try:
        from main import ai_service
        logger.info(f"📥 Đang xử lý yêu cầu lập lịch trình đến: {request.destination} | user: {request.userId or 'anonymous'}")
        
        trip_data = await ai_service.generate_trip(request)

        # ══════════════════════════════════════════════════════════════
        # BẮT BUỘC LƯU TRỮ CHUYẾN ĐI VÀO FIRESTORE (Kể cả anonymous)
        # ══════════════════════════════════════════════════════════════
        user_id = request.userId or getattr(request, "user_id", None)
        trip_id = await ai_service.save_trip_to_db(
            trip_data, 
            user_id,
            owner_name=request.ownerName,
            owner_photo=request.ownerPhoto
        )
        
        if trip_id:
            logger.info(f"✅ [PERSISTENCE] Đã lưu trip_id: {trip_id} cho user: {user_id or 'anonymous'}")

        return StandardResponse(
            status="success",
            message="Lập kế hoạch thành công!",
            data=trip_data
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi lập lịch trình: {e}")
        traceback.print_exc()
        return StandardResponse(
            status="error",
            message=f"Không thể lập lịch trình: {str(e)}",
            data=None
        )

@router.post("/regenerate", response_model=StandardResponse, tags=["AI Trip Planner"])
async def regenerate_trip(payload: Dict[str, Any]):
    """
    Hỗ trợ dịch chuyển hoặc tạo lại lịch trình dựa trên dữ liệu cũ.
    """
    try:
        from main import ai_service
        logger.info("🔄 Nhận yêu cầu tạo lại/dịch chuyển lịch trình")
        trip_data = payload.get("trip_data")
        language_code = payload.get("languageCode", "vi")

        if not trip_data:
            raise HTTPException(status_code=400, detail="Thiếu dữ liệu trip_data để thực hiện.")

        new_trip = await ai_service.regenerate_trip(trip_data, language_code)

        return StandardResponse(
            status="success",
            message=f"Đã dịch chuyển lịch trình sang {language_code}",
            data=new_trip
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi regenerate: {e}")
        return StandardResponse(
            status="error",
            message=str(e),
            data=None
        )

@router.post("/join", response_model=StandardResponse, tags=["Trip Management"])
async def join_trip(request: JoinTripRequest):
    """
    Tham gia vào một chuyến đi đã tồn tại bằng mã chia sẻ (Share Code).
    """
    try:
        from main import ai_service
        if not ai_service._db:
            raise HTTPException(status_code=500, detail="Firestore không khả dụng")

        # 1. Tìm chuyến đi bằng share_code
        docs = ai_service._db.collection("trips").where("share_code", "==", request.share_code).stream()
        trip_list = []
        for doc in docs:
            trip_list.append({"id": doc.id, "data": doc.to_dict()})

        if not trip_list:
            raise HTTPException(status_code=404, detail="Mã mời không hợp lệ")

        # Lấy chuyến đi đầu tiên tìm thấy
        target_trip = trip_list[0]
        doc_id = target_trip["id"]
        trip_data = target_trip["data"]

        # 2. Cập nhật member_ids
        member_ids = trip_data.get("member_ids", [])
        if request.user_id not in member_ids:
            member_ids.append(request.user_id)
            ai_service._db.collection("trips").document(doc_id).update({
                "member_ids": member_ids
            })
            # Cập nhật data để trả về cho người dùng
            trip_data["member_ids"] = member_ids
            logger.info(f"👤 User {request.user_id} đã tham gia trip {doc_id}")
        else:
            logger.info(f"👤 User {request.user_id} đã là thành viên của trip {doc_id}")

        # Trả về kèm thêm ID tài liệu
        trip_data["trip_id"] = doc_id

        return StandardResponse(
            status="success",
            message="Tham gia thành công!",
            data=trip_data
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Lỗi khi tham gia chuyến đi với mã {request.share_code}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống: {str(e)}"
        )
