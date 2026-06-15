from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# ==========================================
# 1. ENUMS & SCHEMAS CHO CHATBOT
# ==========================================

class MessageSender(str, Enum):
    user = "user"
    ai = "ai"

class ChatMessage(BaseModel):
    id: str
    text: str
    sender: MessageSender
    timestamp: datetime

class ChatRequest(BaseModel):
    message: str
    tripContext: Dict[str, Any] = Field(default_factory=dict)
    history: List[ChatMessage] = Field(default_factory=list)

# ==========================================
# 2. SCHEMAS CHO LẬP KẾ HOẠCH DU LỊCH (TRIP PLAN REQUEST)
# ==========================================

class TripPlanRequest(BaseModel):
    """
    Schema cho request tạo lịch trình du lịch AI.

    Hỗ trợ cả 2 tên trường:
      - `userId`  (camelCase) — Flutter app gửi lên
      - `user_id` (snake_case) — backend Python sử dụng nội bộ
    Validator tự động đồng bộ hai chiều để tránh nhầm lẫn.
    """
    # ── Định danh người dùng (dùng cho RAG + Cold Start detection) ──
    userId: Optional[str] = Field(default=None, description="User ID từ Firebase Auth (camelCase, Flutter)")
    user_id: Optional[str] = Field(default=None, description="User ID (snake_case, Python internal)")

    currentLocation: str
    destination: str
    startDate: datetime
    endDate: datetime
    tripType: str
    budget: float
    budgetType: str
    interests: List[str] = Field(default_factory=list)
    companions: str
    accommodationType: str
    transportPreferences: str
    pace: str
    food: str
    languageCode: Optional[str] = "vi"
    mustVisitPlaceIds: Optional[List[str]] = Field(default_factory=list)
    specialRequirements: Optional[str] = None
    
    # ── Thông tin người tạo (dùng để hiển thị trong Group Trips) ──
    ownerName: Optional[str] = None
    ownerPhoto: Optional[str] = None

    @model_validator(mode="after")
    def _sync_user_id_fields(self) -> "TripPlanRequest":
        """
        Đồng bộ userId ↔ user_id sau khi parse:
        - Nếu Flutter gửi userId, gán sang user_id.
        - Nếu backend set user_id, gán ngược lại.
        Ưu tiên userId (camelCase) nếu cả hai đều có giá trị.
        """
        if self.userId and not self.user_id:
            self.user_id = self.userId
        elif self.user_id and not self.userId:
            self.userId = self.user_id
        return self

    model_config = {"populate_by_name": True}


# ==========================================
# 3. SCHEMAS CHO DATABASE (TRAVEL DB MODEL)
# ==========================================

class TravelDbModel(BaseModel):
    travelId: str
    userId: str
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    memberIds: List[str] = Field(default_factory=list)
    shareCode: Optional[str] = None
    destination: str
    destinationLowerCase: str

    currentLocation: Optional[str] = None
    currentLat: Optional[float] = None
    currentLng: Optional[float] = None
    destinationLat: Optional[float] = None
    destinationLng: Optional[float] = None

    startDate: datetime
    endDate: datetime
    overview: Optional[str] = None

    # Sử dụng str thay vì Enum để linh hoạt nhận các giá trị từ Dart (TripType.name)
    tripType: str

    totalDays: int
    totalPeople: int

    images: List[str] = Field(default_factory=list)

    # Tạm thời sử dụng Dict[str, Any] cho các class lồng nhau (DayPlan, v.v.)
    dailyPlan: List[Dict[str, Any]] = Field(default_factory=list)
    accommodation: Dict[str, Any] = Field(default_factory=dict)
    transportationDetails: Dict[str, Any]

    foodRecommendations: List[str] = Field(default_factory=list)
    additionalTips: List[str] = Field(default_factory=list)
    packingList: List[Dict[str, Any]] = Field(default_factory=list)

    budget: Optional[str] = None
    budgetType: Optional[str] = None
    companions: Optional[str] = None
    accommodationType: Optional[str] = None
    transportPreferences: Optional[str] = None
    pace: Optional[str] = None
    foodPreference: Optional[str] = None
    interests: List[str] = Field(default_factory=list)

    isFavorite: bool = False
    language: str
    generatedByAi: bool

    # Sử dụng str cho TripStatus
    status: str

    visitedAt: Optional[datetime] = None
    
    # ── Thông tin định danh chủ sở hữu ──
    ownerName: Optional[str] = None
    ownerPhoto: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

# ==========================================
# 4. SCHEMA RESPONSE CHUẨN (SERVER TRẢ VỀ CHO APP)
# ==========================================

class StandardResponse(BaseModel):
    status: str  # Ví dụ: "success" hoặc "error"
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# ==========================================
# 5. SCHEMA CHO SMART PACKING LIST
# ==========================================

class PackingListRequest(BaseModel):
    destination: str
    days: int
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    weather_context: Optional[str] = "bình thường"
    gender: Optional[str] = "unisex"

# ==========================================
# 6. SCHEMA CHO CO-OP PLANNING
# ==========================================

class JoinTripRequest(BaseModel):
    share_code: str
    user_id: str

# ==========================================
# 7. SCHEMA CHO COLD START PREFERENCES
# ==========================================

class UserPreferencesRequest(BaseModel):
    user_id: str
    interests: List[str]

