import os
import json
import asyncio
import httpx
import re
import logging
import tensorflow as tf
from fastapi import HTTPException
from groq import AsyncGroq
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from models.schemas import TripPlanRequest, ChatRequest
from .ai_prompt_builder import AiPromptBuilder

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AiService:
    """
    Dịch vụ AI tích hợp Groq (Llama-3) cho việc lập lịch trình du lịch và chatbot.
    
    [INTERNAL WIRING – Monolith]
    Hàm generate_trip() giờ đây nhận `user_id` từ request, gọi trực tiếp
    Two-Tower recommendation model (đã load sẵn trong lifespan của main.py),
    lấy danh sách địa điểm cá nhân hóa và chèn vào RAG context trước khi
    gọi Groq, giúp Llama-3 tạo lịch trình phù hợp với sở thích cụ thể của user.
    
    Migrated from trip_ai_backend → triphero_api (Monolith architecture).
    """

    def __init__(self, loaded_model=None, db=None):
        """
        Args:
            loaded_model: TensorFlow Two-Tower SavedModel (được inject từ main.py).
            db: Firestore client (được inject từ main.py).
        """
        # API Keys
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.placeholder_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"

        # Model Groq mạnh nhất hiện tại (Llama-3.3-70b)
        self.model_id = "llama-3.3-70b-versatile"

        # [INTERNAL WIRING] Tham chiếu đến Two-Tower Model và Firestore DB
        # được inject từ lifespan() của main.py
        self._tf_model = loaded_model
        self._db = db

        # Configure Groq Client
        if self.groq_key:
            self.client = AsyncGroq(api_key=self.groq_key)
        else:
            self.client = None
            logger.warning("⚠️  GROQ_API_KEY không tìm thấy. AiService sẽ không hoạt động!")

    # ─── Dependency Injection (gọi sau khi lifespan khởi tạo xong) ──────────────
    def set_dependencies(self, loaded_model, db):
        """Cho phép cập nhật model/db sau khi server khởi động (dùng trong lifespan)."""
        self._tf_model = loaded_model
        self._db = db
        logger.info("✅ AiService: TF Model và Firestore DB đã được inject thành công.")

    # ─── Unsplash Image Fetching ──────────────────────────────────────────────────

    async def _fetch_single_image(self, client: httpx.AsyncClient, query: str) -> str:
        """Lấy URL ảnh landscape từ Unsplash."""
        try:
            trimmed_query = query.strip()
            if not trimmed_query:
                return self.placeholder_url

            if not self.unsplash_key:
                return f"https://source.unsplash.com/1200x675/?{trimmed_query},landscape"

            params = {"query": trimmed_query, "orientation": "landscape", "per_page": "1"}
            headers = {"Authorization": f"Client-ID {self.unsplash_key}"}

            response = await client.get(
                "https://api.unsplash.com/search/photos",
                params=params, headers=headers, timeout=8.0
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results")
                if results and len(results) > 0:
                    urls = results[0].get("urls", {})
                    return urls.get("regular") or urls.get("raw") or None
            else:
                logger.warning(f"⚠️ Unsplash API error {response.status_code}: {response.text}")
                return None

            return None
        except Exception as e:
            logger.error(f"❌ Error fetching image cho '{query}': {e}")
            return None

    async def enrich_images_with_unsplash(self, keywords: List[str]) -> List[str]:
        """Lấy danh sách ảnh từ Unsplash song song (parallel)."""
        if not keywords:
            return []
        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_single_image(client, kw) for kw in keywords]
            results = await asyncio.gather(*tasks)
            # Lọc bỏ các ảnh bị lỗi (None)
            return [url for url in results if url]

    # ─── Groq API Core ────────────────────────────────────────────────────────────

    async def _call_ai_core(
        self,
        prompt: str,
        is_json: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Hàm trung tâm gọi Groq API với cơ chế AUTO-RETRY."""
        if not self.client:
            raise HTTPException(status_code=500, detail="Thiếu GROQ_API_KEY trong cấu hình server.")

        max_retries = 3
        delay_seconds = 2

        # Chuẩn bị messages
        messages = []
        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Đang gọi Groq model: {self.model_id} (Lần thử {attempt + 1}/{max_retries})...")

                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.85,
                    max_tokens=6000,
                    response_format={"type": "json_object"} if is_json else None
                )

                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                logger.error(f"--- LỖI HỆ THỐNG AI (TỪ GROQ) --- Model: {self.model_id} | Error: {error_msg}")

                # Kiểm tra lỗi Rate Limit hoặc Service Unavailable
                if "rate_limit" in error_msg.lower() or "overloaded" in error_msg.lower() or "503" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Groq đang bận. Tự động thử lại sau {delay_seconds} giây...")
                        await asyncio.sleep(delay_seconds)
                        continue
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="Máy chủ Groq hiện đang quá tải. Vui lòng thử lại sau!"
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Lỗi AI ({self.model_id}): {error_msg}"
                    )

    def _clean_json_response(self, text: str) -> str:
        """
        Post-processing pipeline làm sạch response từ Groq trước khi json.loads().

        Xử lý các trường hợp LLM vẫn cố tình sinh ra:
          - Markdown code block: ```json ... ``` hoặc ``` ... ```
          - Văn bản giải thích trước/sau JSON
          - Trailing whitespace / newline thừa
          - BOM character (byte order mark)

        Pipeline 5 bước:
          1. Xóa BOM nếu có
          2. Strip markdown code fences (```json, ```)
          3. Tìm boundary JSON thực sự (bắt đầu từ '{' đầu tiên)
          4. Tìm đóng JSON ('}' cuối cùng tương ứng)
          5. Strip whitespace cuối
        """
        if not text:
            return "{}"

        # ── Bước 1: Xóa BOM và whitespace đầu/cuối ──────────────────────────
        text = text.strip().lstrip('\ufeff')

        # ── Bước 2: Bóc markdown code fence nếu có ───────────────────────────
        # Xử lý: ```json\n{...}\n``` hoặc ```\n{...}\n```
        text = re.sub(
            r'^```(?:json)?\s*\n?',   # opening fence
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r'\n?```\s*$',            # closing fence
            '',
            text,
            flags=re.IGNORECASE
        )
        text = text.strip()

        # ── Bước 3: Tìm vị trí '{' đầu tiên (bỏ qua văn bản trước JSON) ─────
        json_start = text.find('{')
        if json_start == -1:
            logger.error("_clean_json_response: Khong tim thay ky tu '{' trong response.")
            return "{}"
        if json_start > 0:
            logger.warning(
                f"_clean_json_response: Bo qua {json_start} ky tu thua truoc JSON."
            )
            text = text[json_start:]

        # ── Bước 4: Tìm vị trí '}' cuối cùng (bỏ qua văn bản sau JSON) ──────
        json_end = text.rfind('}')
        if json_end == -1:
            logger.error("_clean_json_response: Khong tim thay ky tu '}' trong response.")
            return "{}"
        if json_end < len(text) - 1:
            logger.warning(
                f"_clean_json_response: Bo qua {len(text) - json_end - 1} ky tu thua sau JSON."
            )
            text = text[:json_end + 1]

        # ── Bước 5: Strip lần cuối ────────────────────────────────────────────
        return text.strip()

    # ─── [INTERNAL WIRING] Recommendation Service ────────────────────────────────

    async def get_personalized_places(self, user_id: str) -> List[Dict[str, Any]]:
        """
        [INTERNAL WIRING - Phương thức cầu nối cốt lõi]
        
        Gọi trực tiếp Two-Tower TensorFlow model (đã được load sẵn trong lifespan)
        để lấy danh sách ID địa điểm được cá nhân hóa cho user_id.
        
        Sau đó tra cứu Firestore để lấy thông tin đầy đủ của từng địa điểm,
        trả về list[Dict] chuẩn để AiPromptBuilder.prepare_optimal_rag_context()
        có thể sử dụng.
        
        Returns:
            List[Dict[str, Any]]: Danh sách địa điểm với đầy đủ fields:
                id, name, province, category, latitude, longitude,
                trendingScore, rating, idealDuration
        """
        if not self._tf_model:
            logger.warning("⚠️  Two-Tower model chưa được load. Bỏ qua personalization.")
            return []

        try:
            # 1. Chạy inference Two-Tower Model
            logger.info(f"🔍 Đang lấy recommendations cá nhân hóa cho user: {user_id}")
            input_tensor = tf.constant([user_id])
            scores, destination_ids = self._tf_model(input_tensor)

            # 2. Decode bytes → UTF-8 strings
            raw_ids = destination_ids[0].numpy()
            recommended_ids = [
                dest_id.decode("utf-8") if isinstance(dest_id, bytes) else str(dest_id)
                for dest_id in raw_ids
            ][:20]  # Lấy top 20

            logger.info(f"✅ Two-Tower gợi ý {len(recommended_ids)} địa điểm cho user '{user_id}'")

            # 3. Tra cứu Firestore để lấy thông tin chi tiết
            if not self._db:
                logger.warning("⚠️  Firestore DB chưa kết nối. Trả về raw IDs dạng minimal.")
                # Trả về minimal format để RAG vẫn hoạt động (không có province filter)
                return [{"id": dest_id, "name": dest_id, "province": ""} for dest_id in recommended_ids]

            # Batch get từ Firestore collection 'destinations'
            places = []
            destinations_ref = self._db.collection("destinations")

            for dest_id in recommended_ids:
                try:
                    doc = destinations_ref.document(dest_id).get()
                    if doc.exists:
                        place_data = doc.to_dict()
                        place_data['id'] = dest_id
                        places.append(place_data)
                except Exception as e:
                    logger.warning(f"⚠️  Không thể fetch địa điểm '{dest_id}': {e}")
                    continue

            logger.info(f"📍 Fetch thành công {len(places)}/{len(recommended_ids)} địa điểm từ Firestore.")
            return places

        except Exception as e:
            logger.error(f"❌ Lỗi khi lấy personalized places cho user '{user_id}': {e}")
            return []

    async def get_trending_places(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        [COLD START FALLBACK]
        Lấy danh sách địa điểm phổ biến nhất từ Firestore, sắp xếp theo
        trendingScore giảm dần. Dùng trong 2 trường hợp:
          1. Người dùng ẩn danh (không có user_id).
          2. Người dùng mới — Two-Tower chưa đủ dữ liệu (< 5 địa điểm).

        Args:
            limit: Số địa điểm tối đa cần lấy (mặc định 20).

        Returns:
            List[Dict[str, Any]]: Danh sách địa điểm sorted by trendingScore DESC.
        """
        if not self._db:
            logger.warning("⚠️  Firestore DB chưa kết nối. get_trending_places() trả về rỗng.")
            return []
        try:
            docs = (
                self._db.collection("destinations")
                .order_by("trendingScore", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            places = [{"id": doc.id, **doc.to_dict()} for doc in docs]
            logger.info(f"📈 get_trending_places(): Lấy được {len(places)} địa điểm trending.")
            return places
        except Exception as e:
            # Fallback: Firestore có thể cần index — thử lấy không sort
            logger.warning(f"⚠️  Không thể sort trendingScore (cần Firestore index?): {e}")
            try:
                docs = self._db.collection("destinations").limit(limit).stream()
                return [{"id": doc.id, **doc.to_dict()} for doc in docs]
            except Exception as e2:
                logger.error(f"❌ get_trending_places() hoàn toàn thất bại: {e2}")
                return []

    # ─── Main Business Logic ──────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────
    # RAG SOURCE SELECTION — Ngưỡng phân loại Cold Start
    # ────────────────────────────────────────────────────────────────────────────
    _COLD_START_THRESHOLD = 5
    """
    Số lượng địa điểm tối thiểu Two-Tower cần trả về để coi là "đủ dữ liệu".
    Nếu < ngưỡng này → kích hoạt Cold Start fallback (dùng Trending).
    """

    async def generate_trip(self, request: TripPlanRequest) -> Dict[str, Any]:
        """
        Sinh lịch trình du lịch AI cá nhân hóa dùng Groq (Llama-3).

        ══════════════════════════════════════════════════════════════
        LUỒNG PHÂN LOẠI RAG SOURCE (3 nhánh):
        ══════════════════════════════════════════════════════════════

        [NHÁNH 1 – PERSONALIZED]
          Điều kiện : user_id có giá trị VÀ Two-Tower trả về >= 5 địa điểm
          RAG source : get_personalized_places(user_id) → Two-Tower model
          Log        : 🎯 [PERSONALIZED]

        [NHÁNH 2 – COLD START]
          Điều kiện : user_id có giá trị NHƯNG Two-Tower trả về < 5 địa điểm
                      (người dùng mới, ít lịch sử tương tác)
          RAG source : get_trending_places(limit=20) → Firestore sorted by trendingScore
          Log        : 🧊 [COLD START]

        [NHÁNH 3 – ANONYMOUS]
          Điều kiện : Không có user_id (khách ẩn danh)
          RAG source : get_trending_places(limit=20)
          Log        : 👤 [ANONYMOUS]
        ══════════════════════════════════════════════════════════════
        """
        try:
            # ═══════════════════════════════════════════════════════
            # BƯỚC 1: PHÂN LOẠI NGƯỜI DÙNG & CHỌN NGUỒN RAG
            # ═══════════════════════════════════════════════════════
            user_id = request.user_id or request.userId  # hỗ trợ cả 2 naming convention
            rag_source_label: str  # dùng cho logging
            raw_places: List[Dict[str, Any]]

            if user_id:
                # ── Thử lấy personalized recommendations từ Two-Tower ──
                logger.info(f"🔍 RAG: Đang lấy personalized places cho user '{user_id}'...")
                try:
                    raw_places = await self.get_personalized_places(user_id)
                except Exception as rec_err:
                    logger.warning(f"⚠️ Recommendation engine lỗi: {rec_err}. Kích hoạt Cold Start.")
                    raw_places = []

                if len(raw_places) >= self._COLD_START_THRESHOLD:
                    # ── [NHÁNH 1] PERSONALIZED ──
                    rag_source_label = "PERSONALIZED"
                    logger.info(
                        f"🎯 [PERSONALIZED] user='{user_id}' | "
                        f"{len(raw_places)} địa điểm từ Two-Tower Model."
                    )
                else:
                    # ── [NHÁNH 2] COLD START ──
                    cold_reason = (
                        f"Two-Tower chỉ trả về {len(raw_places)} địa điểm "
                        f"(ngưỡng tối thiểu: {self._COLD_START_THRESHOLD})"
                    )
                    rag_source_label = "COLD START"
                    logger.warning(
                        f"🧊 [COLD START] user='{user_id}' | {cold_reason}. "
                        f"Fallback → get_trending_places()"
                    )
                    raw_places = await self.get_trending_places(limit=20)
            else:
                # ── [NHÁNH 3] ANONYMOUS ──
                rag_source_label = "ANONYMOUS"
                logger.info(
                    "👤 [ANONYMOUS] Không có user_id. "
                    "Sử dụng Trending Places làm RAG context."
                )
                raw_places = await self.get_trending_places(limit=20)

            # ═══════════════════════════════════════════════════════
            # BƯỚC 2: XÂY DỰNG RAG CONTEXT
            # AiPromptBuilder tự filter theo province của request.destination
            # và serialize thành JSON string để bơm vào prompt.
            # ═══════════════════════════════════════════════════════
            rag_context: str = AiPromptBuilder.prepare_optimal_rag_context(request, raw_places)
            logger.info(
                f"📦 RAG Context [{rag_source_label}]: "
                f"{len(raw_places)} places → {len(rag_context)} chars JSON."
            )

            # ═══════════════════════════════════════════════════════
            # BƯỚC 3: BUILD PROMPT VÀ GỌI GROQ
            # ═══════════════════════════════════════════════════════
            prompt = AiPromptBuilder.build_trip_prompt(
                request=request,
                language_code=request.languageCode or "vi",
                available_destinations_json=rag_context
            )

            content = await self._call_ai_core(prompt, is_json=True)

            if not content:
                raise HTTPException(
                    status_code=503,
                    detail="Groq không trả về dữ liệu. Thử lại sau ít giây."
                )

            # ═══════════════════════════════════════════════════════
            # BƯỚC 4: PARSE JSON & ENRICH IMAGES
            # ═══════════════════════════════════════════════════════
            cleaned_text = self._clean_json_response(content)
            try:
                trip_data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                logger.error(f"❌ Lỗi Parse JSON từ Groq: {cleaned_text[:300]}...")
                raise HTTPException(
                    status_code=500,
                    detail="Dữ liệu AI trả về không đúng định dạng JSON."
                )

            keywords = trip_data.get("images", [])
            logger.info(f"📸 Groq generated {len(keywords)} image keywords: {keywords}")

            image_urls = await self.enrich_images_with_unsplash(keywords)
            logger.info(f"✅ Unsplash: fetched {len(image_urls)}/{len(keywords)} images.")

            trip_data["images"] = image_urls
            trip_data["language"] = request.languageCode or "vi"
            trip_data["generatedByAi"] = True
            # Ghi nhận nguồn RAG vào response để frontend/logging có thể theo dõi
            trip_data["_ragSource"] = rag_source_label

            return trip_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"--- LỖI HỆ THỐNG AI (GENERATE) --- {e}")
            raise HTTPException(status_code=500, detail=f"Lỗi không xác định: {str(e)}")

    async def regenerate_trip(self, trip_data: Dict[str, Any], language_code: str) -> Dict[str, Any]:
        """Tạo lại lịch trình hiện có bằng ngôn ngữ khác dùng Groq."""
        try:
            prompt = AiPromptBuilder.build_trip_regeneration_prompt(trip_data, language_code)

            content = await self._call_ai_core(prompt, is_json=True)

            if not content:
                raise HTTPException(status_code=503, detail="AI không phản hồi khi dịch chuyển.")

            cleaned_text = self._clean_json_response(content)
            new_trip_data = json.loads(cleaned_text)

            new_trip_data["images"] = trip_data.get("images", [])
            new_trip_data["language"] = language_code

            return new_trip_data
        except Exception as e:
            logger.error(f"❌ Lỗi khi dịch chuyển lịch trình: {e}")
            raise HTTPException(status_code=500, detail=f"Lỗi dịch chuyển: {str(e)}")

    async def chat_with_ai(self, request: ChatRequest) -> str:
        """Xử lý tin nhắn chatbot dùng Groq."""
        try:
            user_input = AiPromptBuilder.build_chat_prompt(
                message=request.message,
                trip_context=request.tripContext
            )

            # Chuẩn bị lịch sử theo format Groq (OpenAI style)
            history = []
            for msg in request.history:
                role = "user" if msg.sender == "user" else "assistant"
                history.append({"role": role, "content": msg.text})

            reply = await self._call_ai_core(user_input, is_json=False, chat_history=history)

            return reply

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"--- LỖI HỆ THỐNG AI (CHAT) --- {e}")
            raise HTTPException(status_code=500, detail="Không thể gửi tin nhắn lúc này.")
