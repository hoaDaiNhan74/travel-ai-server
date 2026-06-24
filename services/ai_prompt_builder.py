import json
import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import từ package models của triphero_api (đường dẫn đã được cập nhật)
from models.schemas import TripPlanRequest, PackingListRequest


class AiPromptBuilder:
    """
    Class to build prompts for Groq AI (Llama-3).
    Optimized following user's specific logic (Iron Rules & RAG).
    Migrated from trip_ai_backend → triphero_api (Monolith).
    """

    # ---------------------------------------------------------
    # 1. HÀM CHUẨN HÓA CHUỖI TIẾNG VIỆT (Tương đương _normalize)
    # ---------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        # Loại bỏ dấu tiếng Việt
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        # Đổi 'đ' thành 'd' và đưa về chữ thường
        return text.replace('đ', 'd').replace('Đ', 'D').lower()

    # ---------------------------------------------------------
    # 2. TÍNH TOÁN SỐ NGƯỜI (Tương đương _getPeopleCount)
    # ---------------------------------------------------------
    @staticmethod
    def _get_people_count(companions: str) -> int:
        comp = companions.lower()
        if 'solo' in comp: return 1
        if 'partner' in comp or 'couple' in comp: return 2
        if 'family' in comp: return 4
        if 'friends' in comp: return 3
        return 1

    # ---------------------------------------------------------
    # 3. CHUẨN BỊ NGỮ CẢNH RAG TỐI ƯU
    # all_destinations: dữ liệu từ DB (Firestore) hoặc danh sách recommend từ Two-Tower model
    # ---------------------------------------------------------
    @staticmethod
    def prepare_optimal_rag_context(request: TripPlanRequest, all_destinations: List[Dict[str, Any]] = []) -> str:
        try:
            destination_lower = AiPromptBuilder._normalize(request.destination)

            # 1. Lọc destinationPool: Tìm các địa điểm thuộc tỉnh/thành phố yêu cầu
            destination_pool = []
            for d in all_destinations:
                prov = AiPromptBuilder._normalize(d.get('province', ''))
                if prov and (destination_lower in prov or prov in destination_lower):
                    destination_pool.append(d)

            if not destination_pool:
                return "{}"

            # 2. Tách dữ liệu: Must visit vs Suggestions
            must_visit_ids = request.mustVisitPlaceIds or []
            must_visit_places = [d for d in all_destinations if d.get('id') in must_visit_ids]
            suggestion_pool = [d for d in destination_pool if d.get('id') not in must_visit_ids]

            # Sort theo trendingScore giảm dần, sau đó là rating
            suggestion_pool.sort(key=lambda x: (x.get('trendingScore', 0), x.get('rating', 0)), reverse=True)
            top_suggestions = suggestion_pool[:20]

            # 3. Đóng gói JSON mỏng nhẹ
            def map_to_minimal(d: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    'id': d.get('id'),
                    'name': d.get('name'),
                    'category': d.get('category', 'Attraction'),
                    'lat': d.get('latitude'),
                    'lng': d.get('longitude'),
                    'idealDuration': d.get('idealDuration', 2)
                }

            result = {
                'user_must_visit': [map_to_minimal(d) for d in must_visit_places],
                'ai_suggestion_pool': [map_to_minimal(d) for d in top_suggestions]
            }
            return json.dumps(result, ensure_ascii=False)

        except Exception:
            return "{}"

    # ---------------------------------------------------------
    # 4. BUILD HỆ THỐNG PROMPT TẠO LỊCH TRÌNH
    # ---------------------------------------------------------
    @staticmethod
    def build_trip_prompt(
        request: TripPlanRequest, 
        language_code: str = "vi", 
        available_destinations_json: str = "{}",
        weather_context: str = "",
        events_context: str = ""
    ) -> str:

        # ── Few-shot JSON schema mẫu (LLM phải bắt chước chính xác) ──────────
        # Quy tắc Emoji được thể hiện ngay trong ví dụ này:
        #   - Emoji ĐẦU TIÊN trong title/description, không lặp lại.
        #   - Không emoji ở cuối câu hay giữa đoạn văn.
        json_schema_example = """{
  "destination": "Đà Nẵng",
  "currentLocation": "Hà Nội",
  "currentLat": 21.0285,
  "currentLng": 105.8542,
  "destinationLat": 16.0544,
  "destinationLng": 108.2022,
  "startDate": "2025-07-01T00:00:00",
  "endDate": "2025-07-03T00:00:00",
  "overview": "Chuyến hành trình 3 ngày tại Đà Nẵng mang đến những trải nghiệm khó quên bên bờ biển xanh trong. Thành phố đáng sống này hội tụ di sản văn hóa Chăm Pa ngàn năm, những cây cầu mang tầm vóc kiến trúc quốc tế và nền ẩm thực địa phương phong phú. Mỗi buổi sáng bắt đầu bằng làn gió mát từ biển Mỹ Khê, mỗi chiều tà được nhuộm vàng bởi hoàng hôn trên cầu Rồng rực rỡ.",
  "tripType": "Beach",
  "totalDays": 3,
  "totalPeople": 2,
  "images": [
    "golden bridge ba na hills danang",
    "dragon bridge da nang night",
    "my khe beach sunrise",
    "marble mountains da nang",
    "son tra peninsula nature"
  ],
  "dailyPlan": [
    {
      "day": 1,
      "date": "2025-07-01",
      "activities": [
        {
          "time": "08:00",
          "title": "🍜 Ăn sáng đặc sản",
          "destinationId": null,
          "description": "Thưởng thức món ăn đặc trưng của địa phương để bắt đầu ngày mới đầy năng lượng. Hương vị đậm đà cùng cách chế biến truyền thống sẽ mang đến trải nghiệm ẩm thực khó quên ngay từ buổi sáng đầu tiên.",
          "type": "Indoor",
          "reasoning": "Thời tiết sáng mát mẻ nhưng bạn muốn dùng bữa trong không gian truyền thống để nạp năng lượng."
        }
      ]
    },
    {
      "day": 2,
      "date": "2025-07-02",
      "activities": [
        {
          "time": "09:00",
          "title": "🏛️ Tham quan bảo tàng",
          "destinationId": "museum_id",
          "description": "Khám phá những hiện vật lịch sử và văn hóa đặc sắc được lưu giữ qua nhiều thế hệ. Đây là cơ hội tuyệt vời để hiểu sâu hơn về cội nguồn và những câu chuyện thú vị đằng sau các di sản quý giá.",
          "type": "Indoor",
          "reasoning": "Dự báo có mưa rào nhẹ vào buổi sáng nên ưu tiên các hoạt động trong nhà."
        },
        {
          "time": "19:00",
          "title": "🎆 Tham gia Lễ hội ánh sáng",
          "destinationId": "event_square",
          "description": "Hòa mình vào không khí sôi động của Lễ hội ánh sáng với nhiều tiết mục biểu diễn độc đáo và các gian hàng ẩm thực hấp dẫn.",
          "type": "Event",
          "reasoning": "Đang có sự kiện đặc biệt diễn ra tại điểm đến vào đúng thời gian bạn lưu trú."
        }
      ]
    }
  ],
  "accommodationSuggestions": [
    {
      "name": "Mường Thanh Luxury Đà Nẵng",
      "type": "Hotel",
      "location": "Quận Sơn Trà, Đà Nẵng",
      "priceRange": "1.200.000 - 2.500.000 VND/đêm"
    },
    {
      "name": "Homie Hostel Da Nang",
      "type": "Hostel",
      "location": "Quận Hải Châu, Đà Nẵng",
      "priceRange": "150.000 - 350.000 VND/đêm"
    }
  ],
  "transportationDetails": {
    "localTransport": "Xe máy thuê hoặc Grab là lựa chọn tối ưu để di chuyển trong thành phố với chi phí hợp lý.",
    "tips": "Tránh di chuyển bằng xe máy vào khung giờ 11:00-13:00 mùa hè do nắng gắt. Grab bike thường rẻ hơn 30% so với Grab car."
  },
  "foodRecommendations": [
    "🍜 Bún chả cá - đặc sản buổi sáng không thể bỏ qua",
    "🦞 Mì Quảng tôm thịt - hương vị đậm đà miền Trung",
    "🥩 Bánh mì Phượng - thương hiệu nổi tiếng thế giới"
  ],
  "additionalTips": [
    "📱 Tải app Grab trước khi đến để tiện di chuyển",
    "🕑 Đặt vé cáp treo Bà Nà Hills trước ít nhất 1 ngày vào mùa cao điểm",
    "🌊 Tránh tắm biển khi có cờ đỏ — tuân thủ tuyệt đối"
  ],
  "budget": "5.000.000 VND - 8.000.000 VND cho 2 người trong 3 ngày"
}"""

        prompt = f"""Bạn là một CHUYÊN GIA DU LỊCH ĐỊA PHƯƠNG chuyên nghiệp và am hiểu sâu sắc về văn hóa, ẩm thực, địa lý Việt Nam.

════════════════════════════════════════════════════════
NGÔN NGỮ BẮT BUỘC
════════════════════════════════════════════════════════
- Toàn bộ nội dung trả lời BẮT BUỘC bằng Tiếng Việt (language_code: "{language_code}").
- Tên địa danh giữ nguyên bản gốc. Mô tả, cảm nhận PHẢI dịch sang tiếng Việt.
- TUYỆT ĐỐI KHÔNG trộn lẫn ngôn ngữ khác vào nội dung.

════════════════════════════════════════════════════════
QUY TẮC ĐỊNH DẠNG ĐẦU RA — STRICT (VI PHẠM = LỖI NGHIÊM TRỌNG)
════════════════════════════════════════════════════════
1. Trả về MỘT chuỗi JSON hợp lệ duy nhất. Không có gì khác.
2. TUYỆT ĐỐI KHÔNG dùng markdown code block (``` hoặc ```json).
3. TUYỆT ĐỐI KHÔNG thêm lời chào hỏi, giải thích, hay bất kỳ văn bản nào trước hoặc sau JSON.
4. TUYỆT ĐỐI KHÔNG để các trường có giá trị null (ngoại trừ trường "destinationId").
   Nếu không có dữ liệu → dùng "" (chuỗi rỗng), 0 (số), hoặc [] (mảng rỗng).
5. Tên trường (key) phải KHỚP CHÍNH XÁC với schema. Không thêm, không xóa, không đổi tên.
6. Tọa độ (lat/lng) phải là số thực hợp lệ, phản ánh đúng vị trí địa lý thực tế.

════════════════════════════════════════════════════════
QUY TẮC EMOJI — STRICT (VI PHẠM = LỖI NGHIÊM TRỌNG)
════════════════════════════════════════════════════════
1. Emoji CHỈ được xuất hiện ở VỊ TRÍ ĐẦU TIÊN của chuỗi (title, foodRecommendations item, additionalTips item).
   ✅ ĐÚNG: "🍜 Ăn sáng bún bò Huế đặc sản"
   ✅ ĐÚNG: "🏖️ Tắm biển Mỹ Khê"
   ❌ SAI:  "Ăn sáng bún bò Huế 🍜 ngon tuyệt"
   ❌ SAI:  "Tắm biển Mỹ Khê, trải nghiệm sóng biển 🌊 thú vị 🏖️"
2. Mỗi mục CHỈ dùng TỐI ĐA MỘT Emoji.
3. Trường "description" và "overview" TUYỆT ĐỐI KHÔNG chứa Emoji — chỉ dùng văn bản thuần túy.
4. Trường "transportationDetails", "accommodationSuggestions" KHÔNG dùng Emoji.

════════════════════════════════════════════════════════
YÊU CẦU CHẤT LƯỢNG NỘI DUNG — BẮT BUỘC
════════════════════════════════════════════════════════
- "overview": Tối thiểu 4-5 câu, miêu tả sinh động cảnh quan, cảm xúc và điểm nổi bật của chuyến đi.
- "description" trong mỗi activity: Tối thiểu 3-4 câu, đề cập lịch sử, kiến trúc, văn hóa, hoặc trải nghiệm đặc biệt.
════════════════════════════════════════════════════════
QUY TẮC LỊCH TRÌNH (DAILY PLAN) — QUAN TRỌNG NHẤT
════════════════════════════════════════════════════════
1. Mảng "dailyPlan" BẮT BUỘC phải chứa ĐÚNG {(request.endDate - request.startDate).days + 1} phần tử (tương ứng từ Ngày 1 đến Ngày {(request.endDate - request.startDate).days + 1}).
2. TUYỆT ĐỐI KHÔNG được tóm tắt, không được bỏ sót bất kỳ ngày nào ở giữa.
3. Mỗi ngày PHẢI có ít nhất 3-5 hoạt động (Sáng, Trưa, Chiều, Tối).
4. Nếu chuyến đi dài ngày (ví dụ 9-10 ngày), hãy đảm bảo phân bổ sức lực hợp lý (xen kẽ ngày khám phá mạnh và ngày nghỉ ngơi nhẹ nhàng).

════════════════════════════════════════════════════════
SCHEMA JSON MẪU — BẮT CHƯỚC CHÍNH XÁC CẤU TRÚC NÀY
════════════════════════════════════════════════════════
{json_schema_example}

════════════════════════════════════════════════════════
THÔNG TIN CHUYẾN ĐI
════════════════════════════════════════════════════════
- currentLocation    : {request.currentLocation}
- destination        : {request.destination}
- startDate          : {request.startDate.isoformat()}
- endDate            : {request.endDate.isoformat()}
- tripType           : {request.tripType}
- budget             : {request.budget} ({request.budgetType})
- interests          : {', '.join(request.interests)}
- companions         : {request.companions}
- accommodationType  : {request.accommodationType}
- transportPreferences: {request.transportPreferences}
- pace               : {request.pace}
- foodPreferences    : {request.food}
- specialRequirements: {request.specialRequirements or 'Không có'}
- mustVisitPlaceIds  : {', '.join(request.mustVisitPlaceIds) if request.mustVisitPlaceIds else 'Không có'}
- totalPeople        : {AiPromptBuilder._get_people_count(request.companions)}

════════════════════════════════════════════════════════
DỮ LIỆU ĐỊA ĐIỂM (RAG CONTEXT — ƯU TIÊN TUYỆT ĐỐI)
════════════════════════════════════════════════════════
{available_destinations_json}

════════════════════════════════════════════════════════
NGỮ CẢNH THỜI TIẾT & SỰ KIỆN (CONTEXT-AWARE RAG)
════════════════════════════════════════════════════════
- Thời tiết: {weather_context if weather_context else 'Không có thông tin dự báo.'}
- Sự kiện: {events_context if events_context else 'Không có sự kiện đặc biệt nào.'}

NGUYÊN TẮC XỬ LÝ NGỮ CẢNH & DỮ LIỆU:
1. NẾU dự báo có mưa/nắng gắt → BẮT BUỘC ưu tiên các địa điểm trong nhà (Bảo tàng, Quán cafe, Chùa...) và gán "type": "Indoor".
2. NẾU thời tiết đẹp → Ưu tiên các hoạt động ngoài trời, dã ngoại, tắm biển... và gán "type": "Outdoor".
3. NẾU có sự kiện/lễ hội đang diễn ra → Cố gắng chèn sự kiện đó vào lịch trình (phù hợp thời gian) và gán "type": "Event".
4. NẾU KHÔNG CÓ sự kiện nào được cung cấp trong ngữ cảnh → TUYỆT ĐỐI KHÔNG tự bịa ra các lễ hội/sự kiện ảo.
5. BẮT BUỘC cung cấp "reasoning" (lý do) cho MỖI hoạt động, giải thích tại sao chọn địa điểm/hoạt động đó dựa trên THỜI TIẾT, SỰ KIỆN hoặc SỞ THÍCH của người dùng.
6. ƯU TIÊN lấy địa điểm từ database trên để lập lịch, không tự bịa đặt. Gán "destinationId" tương ứng.
7. "mustVisitPlaceIds" BẮT BUỘC xuất hiện trong itinerary.
8. Tuân thủ tuyệt đối "specialRequirements" nếu có.

════════════════════════════════════════════════════════
DERIVED FIELDS (BẮT BUỘC TÍNH ĐÚNG)
════════════════════════════════════════════════════════
- totalDays   = {(request.endDate - request.startDate).days + 1} (Bạn PHẢI tạo lịch trình cho ĐÚNG {(request.endDate - request.startDate).days + 1} ngày)
- totalPeople = {AiPromptBuilder._get_people_count(request.companions)}

════════════════════════════════════════════════════════
QUY TẮC TRƯỜNG "images" (STRICT)
════════════════════════════════════════════════════════
- PHẢI là array chứa ĐÚNG 5 phần tử.
- Mỗi phần tử là từ khóa tìm kiếm ảnh bằng TIẾNG ANH, chữ thường, không dấu.
- Bắt buộc kết hợp tên điểm đến vào từ khóa (ví dụ: "dragon bridge da nang").
- Tối thiểu: 1 cảnh thiên nhiên, 1 công trình/kiến trúc, 1 toàn cảnh thành phố.
- KHÔNG trả về URL, link, hay markdown.

Bây giờ hãy tạo lịch trình ĐẦY ĐỦ cho {(request.endDate - request.startDate).days + 1} NGÀY tại {request.destination}.
Trả về NGAY JSON object — không có bất kỳ văn bản nào khác.
"""
        return prompt

    @staticmethod
    def build_trip_regeneration_prompt(trip: Dict[str, Any], language_code: str) -> str:
        """
        Builds the prompt for regenerating a trip in a different language.
        """
        return f'''
You are an AI travel planner.

TASK:
Regenerate the SAME trip in a different language.

CRITICAL LANGUAGE RULE:
- Rewrite ALL user-facing text strictly in "{language_code}"
- Do NOT change meaning, structure, or plan
- Do NOT invent new places, days, or activities
- Keep dates, coordinates, counts EXACTLY the same

CRITICAL FORMAT RULES:
- Return ONLY a valid JSON object
- Do NOT include markdown, comments, or explanations

EXISTING TRIP DATA (REFERENCE ONLY):
{json.dumps(trip, ensure_ascii=False)}

Return ONLY the JSON object.
'''

    # ---------------------------------------------------------
    # 5. BUILD PROMPT CHO CHATBOT
    # ---------------------------------------------------------
    @staticmethod
    def build_chat_prompt(message: str, trip_context: dict) -> str:
        destination = trip_context.get('destination', 'Không rõ')
        start_date = trip_context.get('startDate', 'Không rõ')
        end_date = trip_context.get('endDate', 'Không rõ')
        overview = trip_context.get('overview', 'Không có mô tả')

        system_prompt = f"""
Bạn là một Trợ lý Du lịch AI thân thiện và chuyên nghiệp.
Bạn đang hỗ trợ người dùng trong chuyến đi của họ. Dưới đây là thông tin về chuyến đi hiện tại:
- Điểm đến: {destination}
- Ngày đi: {start_date}
        Hãy trả lời các câu hỏi của người dùng ngắn gọn, súc tích, bằng tiếng Việt. Nếu họ hỏi những thứ không liên quan đến du lịch, hãy lịch sự từ chối và hướng họ về chuyến đi.

        Người dùng hỏi: {message}
        """
        return system_prompt

    # ---------------------------------------------------------
    # 6. BUILD PROMPT CHO SMART PACKING LIST
    # ---------------------------------------------------------
    @staticmethod
    def build_packing_list_prompt(request: PackingListRequest) -> str:
        prompt = f"""Bạn là một chuyên gia hậu cần du lịch. Hãy lập danh sách hành lý cho chuyến đi {request.days} ngày đến {request.destination}. Giới tính: {request.gender}.
Ngữ cảnh: {request.weather_context}

QUY TẮC EMOJI: MỖI danh mục BẮT BUỘC bắt đầu bằng MỘT Emoji phù hợp (VD: '👔 Quần áo'). TUYỆT ĐỐI KHÔNG thêm Emoji vào tên các món đồ bên trong (VD: Chỉ ghi 'Giày thể thao' thay vì '👟 Giày thể thao').
QUY TẮC SUY LUẬN (REASONING): Nếu món đồ được đề xuất ĐẶC BIỆT dựa trên NGỮ CẢNH (thời tiết hoặc sự kiện), bạn BẮT BUỘC cung cấp "reasoning" giải thích ngắn gọn tại sao. Với các món đồ cơ bản (bàn chải, sạc, áo thun...), để reasoning là null.
QUY TẮC ĐỊNH DẠNG: BẮT BUỘC trả về JSON thuần túy (Raw JSON). TUYỆT ĐỐI KHÔNG dùng markdown block (như ```json). Không giải thích thêm.

Cấu trúc JSON yêu cầu (Few-shot example):
{{
  "categories": [
    {{
      "name": "👔 Quần áo",
      "items": [
        {{
          "name": "Áo khoác chống nước",
          "reasoning": "Dự báo có mưa rào nhẹ vào buổi chiều nên cần áo khoác chống nước."
        }},
        {{
          "name": "Áo thun",
          "reasoning": null
        }}
      ]
    }}
  ]
}}

TRẢ VỀ JSON NGAY BÂY GIỜ:"""
        return prompt
