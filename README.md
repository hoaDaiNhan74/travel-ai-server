# 🧭 TripHero Unified API (Trekko Backend)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Groq AI](https://img.shields.io/badge/Groq%20AI-F34F29?style=for-the-badge)](https://console.groq.com)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**TripHero Unified API (Trekko Backend)** là hệ thống máy chủ dịch vụ (Monolith API) tích hợp trí tuệ nhân tạo (Generative AI) và học máy (Machine Learning) để cung cấp giải pháp du lịch thông minh, cá nhân hóa. Đây là backend cốt lõi hỗ trợ trực tiếp cho ứng dụng di động **TripHero** (viết bằng Flutter) và hệ thống quản trị đối tác.

---

## 📌 Tổng Quan Về Hệ Thống
Hệ thống được thiết kế dạng Monolith hiệu năng cao dựa trên nền tảng **FastAPI**, kết hợp ba khối xử lý thông minh chính:
1. **Engine Gợi ý Địa điểm (Two-Tower ML Model)**: Mô hình học máy dựa trên mạng neural của TensorFlow Recommenders để đề xuất địa điểm du lịch tối ưu cho từng người dùng cụ thể.
2. **Trợ lý AI Lập kế hoạch & Chatbot (Groq AI - Llama-3)**: Tự động thiết kế lịch trình du lịch chi tiết và hỗ trợ trò chuyện thông minh theo ngữ cảnh chuyến đi.
3. **Phân tích Hành vi & Phân khúc (Scikit-Learn K-Means)**: Tự động gom cụm người dùng dựa trên thông số nhân khẩu học, mức chi tiêu và mức độ tương tác.

### 🌟 Tính năng chính:
- **Gợi ý Cá nhân hóa (Personalized Recommendations)**: API [get_recommendations](file:///D:/DACN2/triphero_api/main.py#L176-L218) trả về danh sách top 10 địa điểm du lịch phù hợp dựa trên mô hình Two-Tower.
- **RAG-Augmented AI Trip Planner**: API [generate_trip](file:///D:/DACN2/triphero_api/routers/trip_router.py#L71-L111) sinh lịch trình du lịch chi tiết dạng JSON, tự động nạp danh sách địa điểm gợi ý từ Two-Tower model hoặc địa điểm nổi bật (Trending Places) làm ngữ cảnh (Retrieval-Augmented Generation - RAG).
- **Trợ lý ảo thông minh (AI Chatbot)**: API [chat_with_ai](file:///D:/DACN2/triphero_api/main.py#L237-L260) hỗ trợ giải đáp mọi thắc mắc du lịch của người dùng, ghi nhớ lịch sử hội thoại và ngữ cảnh chuyến đi hiện tại.
- **Tạo danh sách hành lý (Packing List Generator)**: API [get_packing_list](file:///D:/DACN2/triphero_api/routers/ai_router.py#L7-L20) gợi ý đồ dùng cần chuẩn bị dựa theo thời tiết và sự kiện tại điểm đến.
- **Phân tích & Phân khúc Khách hàng**: Định kỳ phân tích dữ liệu Firestore để dán nhãn phân khúc người dùng (*Luxury Elites, Active Budget Backpackers, Standard Vacationers, Occasional/Churned*) và xuất biểu đồ phân tích trực quan.
- **Hệ thống Tự động Huấn luyện (Retrain Pipeline)**: Hỗ trợ tự động chạy ngầm tải tương tác người dùng mới nhất từ Firestore để huấn luyện lại mô hình TensorFlow, tự động backup và nạp lại mô hình mà không cần dừng server.

---

## 🛠️ Công Nghệ Sử Dụng

Hệ thống được xây dựng trên nền tảng Python hiện đại với các thư viện chuyên dụng sau được cấu hình trong [requirements.txt](file:///D:/DACN2/triphero_api/requirements.txt):

| Danh mục | Công nghệ / Thư viện | Vai trò |
| :--- | :--- | :--- |
| **Core Web API** | [FastAPI](https://fastapi.tiangolo.com) | Khung phát triển API hiệu năng cao, bất đồng bộ (Asynchronous) |
| | [Uvicorn](https://www.uvicorn.org) & [Gunicorn](https://gunicorn.org) | ASGI web server chạy API trong môi trường phát triển và sản xuất |
| **Machine Learning** | [TensorFlow](https://www.tensorflow.org) & [TF-Keras](https://github.com/tensorflow/keras) | Xây dựng và tính toán mạng neural |
| | [TensorFlow Recommenders (TFRS)](https://www.tensorflow.org/recommenders) | Xây dựng kiến trúc Two-Tower Retrieval Model |
| **Data Analytics** | [scikit-learn](https://scikit-learn.org) | Huấn luyện mô hình phân cụm K-Means & Giảm chiều dữ liệu PCA |
| | [pandas](https://pandas.pydata.org) & [numpy](https://numpy.org) | Biến đổi, tiền xử lý và kỹ nghệ đặc trưng (Feature Engineering) |
| | [matplotlib](https://matplotlib.org) | Vẽ các biểu đồ phân tích và lưu trữ dưới dạng ảnh tĩnh tĩnh |
| **Generative AI** | [Groq SDK](https://console.groq.com) (Llama-3.3-70b-versatile) | LLM siêu nhanh để sinh nội dung lịch trình, trả lời chatbot, hành lý |
| **Cloud Services** | [Firebase Admin SDK](https://firebase.google.com/docs/admin) | Kết nối, đọc/ghi cơ sở dữ liệu NoSQL Firestore thời gian thực |
| | [Unsplash API](https://unsplash.com/developers) | Tìm kiếm song song (Parallel fetching) ảnh bìa phong cảnh chất lượng cao |
| **Utilities** | [APScheduler](https://apscheduler.readthedocs.io) | Quản lý và lên lịch các tác vụ chạy ngầm định kỳ |
| | [python-dotenv](https://github.com/theofidry/django-dotenv) | Quản lý cấu hình biến môi trường an toàn |

---

## 📐 Kiến Trúc & Luồng Hoạt Động

Dưới đây là sơ đồ luồng hoạt động tổng quan của hệ thống:

```mermaid
graph TD
    User([User App / Admin]) -->|1. Requests /api/v1/recommend/{user_id}| FastAPI[FastAPI Server]
    FastAPI -->|2. Convert user_id to Tensor| TF[Two-Tower TensorFlow Model]
    TF -->|3. Get recommended Destination IDs| FastAPI
    
    User -->|4. Requests /api/v1/trips/generate| FastAPI
    FastAPI -->|5. Get recommended or trending places| DB[(Cloud Firestore)]
    FastAPI -->|6. Call weather & events APIs| Context[Weather & Events Mock API]
    FastAPI -->|7. Combine data into Prompt| Groq[Groq AI Llama-3]
    Groq -->|8. Structured JSON Itinerary| FastAPI
    FastAPI -->|9. Enrich with cover images| Unsplash[Unsplash API]
    FastAPI -->|10. Save trip details| DB
    FastAPI -->|11. Return detailed Trip Plan| User

    Admin([Admin / Cron Job]) -->|12. Trigger retrain /train| Analytics[Analytics Pipeline]
    Analytics -->|13. Fetch profiles, trips, expenses| DB
    Analytics -->|14. Run K-Means Clustering| SK[Scikit-Learn K-Means]
    SK -->|15. Update segment labels| DB
    Analytics -->|16. Export demographics & clusters| Charts[uploads/charts/ Demographics & PCA Plots]

    User -->|17. Log User Interactions| DB
    Admin -->|18. Trigger retrain /admin/retrain| Retrain[Two-Tower Retrain Pipeline]
    Retrain -->|19. Fetch 30-day interactions & destinations| DB
    Retrain -->|20. Train model for 15 epochs| TF
```

### 1. Luồng Gợi ý Địa điểm (Two-Tower Recommendation)
- Mô hình gồm 2 nhánh mạng neural:
  - **User Tower**: Nhận đầu vào là `user_id` để xuất ra vector embedding biểu diễn sở thích người dùng.
  - **Destination Tower**: Nhận đầu vào là `destination_id`, `category` (thể loại) và `tags` (nhãn đặc trưng đã được Vectorize) để xuất ra vector embedding biểu diễn đặc tính địa điểm.
- Khi gọi API `/api/v1/recommend/{user_id}`:
  - Server truyền `user_id` vào mô hình `BruteForce` đã được biên dịch sẵn.
  - Mô hình thực hiện nhân vô hướng (Dot Product) vector embedding của user với toàn bộ vector địa điểm trong kho dữ liệu để tính điểm số tương thích.
  - Trả về top 10 địa điểm có điểm số cao nhất dưới dạng danh sách IDs.

### 2. Luồng Sinh Lịch Trình (RAG-Augmented Planner)
- Khi nhận yêu cầu lập lịch trình qua `/api/v1/trips/generate`, server phân loại người dùng để nạp dữ liệu RAG tối ưu thông qua lớp [AiService](file:///D:/DACN2/triphero_api/services/ai_service.py#L26-L629):
  - **Nhánh Personalized**: Nếu người dùng đã đăng nhập và có lịch sử tương tác dày dặn, server gọi Two-Tower model để lấy danh sách địa điểm cá nhân hóa.
  - **Nhánh Cold Start**: Nếu người dùng đã đăng nhập nhưng ít tương tác, hệ thống tự động fallback lấy danh sách địa điểm thịnh hành (`trendingScore` cao nhất) từ Firestore.
  - **Nhánh Anonymous**: Khách ẩn danh được đề xuất địa điểm thịnh hành chung.
- Dữ liệu địa điểm (RAG Context) kết hợp cùng bối cảnh thời tiết & sự kiện được nạp vào Prompt Builder để tạo ra System/User Prompt chuẩn hóa.
- Server gửi prompt tới Groq API (Llama-3.3-70b-versatile) để lấy kết quả lịch trình dạng JSON thô.
- Server thực hiện lọc, dọn dẹp JSON phản hồi (`_clean_json_response`), đồng thời gọi song song Unsplash API để nạp ảnh bìa chất lượng cao.
- Toàn bộ thông tin chuyến đi được lưu vào Firestore dưới dạng tài liệu chuyến đi (`trips`), tạo mã chia sẻ (`share_code`) phục vụ việc mời thành viên cùng tham gia qua endpoint [join_trip](file:///D:/DACN2/triphero_api/routers/trip_router.py#L143-L196).

### 3. Luồng Phân tích & Phân khúc Người dùng (K-Means Clustering)
- Khi kích hoạt huấn luyện phân cụm qua endpoint [trigger_analytics_train](file:///D:/DACN2/triphero_api/routers/analytics_router.py#L61-L72):
  - Kéo toàn bộ dữ liệu người dùng, chuyến đi và hóa đơn chi tiêu (`expenses`) từ Firestore.
  - Trích xuất 6 đặc trưng chính: *tuổi, thời gian chuyến đi trung bình, tổng chi tiêu du lịch, tổng số tương tác/hoạt động, thời lượng phiên truy cập, số lượng chuyến đi*.
  - Tiền xử lý chuẩn hóa dữ liệu (`StandardScaler`) và đưa vào mô hình phân cụm **K-Means (K=4)**.
  - Ánh xạ nhãn cụm sang 4 nhóm khách hàng mục tiêu: *Luxury Elites, Active Budget Backpackers, Standard Vacationers, Occasional/Churned*.
  - Thực hiện cập nhật hàng loạt (Batch Update) nhãn phân khúc ngược lại Firestore của từng người dùng.
  - Sử dụng PCA giảm chiều xuống không gian 2D để vẽ biểu đồ phân cụm, kết xuất cùng các biểu đồ nhân khẩu học, chi tiêu du lịch, và lưu vào thư mục static `/uploads/charts/` để client hiển thị trực tiếp.

### 4. Luồng Tự động Huấn luyện lại (Retrain Pipeline)
- Mô hình Two-Tower tự động thu thập các loại tương tác người dùng trên app: *view_detail (1.0đ), view_map (1.5đ), dwell_time (2.0đ), share_destination (3.0đ), add_favorite (4.0đ), rate_destination (5.0đ), plan_trip (5.0đ)*.
- Khi chạy pipeline retrain qua hàm [run_retrain_pipeline](file:///D:/DACN2/triphero_api/scripts/retrain_pipeline.py#L350-L371):
  - Tải dữ liệu tương tác trong vòng 30 ngày qua và toàn bộ danh sách địa điểm.
  - Nếu số lượng tương tác hợp lệ >= 50, tiến hành huấn luyện lại mạng neural Two-Tower qua 15 epochs.
  - Biên dịch và cập nhật lại lớp tìm kiếm BruteForce Index.
  - Tự động sao lưu phiên bản cũ vào thư mục backup, ghi đè phiên bản mới vào thư mục chính và lưu thông tin metadata huấn luyện (`retrain_metadata.json`).
  - Nếu xảy ra lỗi trong quá trình lưu mô hình mới, hệ thống tự động khôi phục lại bản sao lưu gần nhất.

---

## 🏃 Hướng Dẫn Cài Đặt & Chạy Server

### 📋 Yêu cầu hệ thống
- **Python**: Phiên bản **3.9 đến 3.11** (Bắt buộc, để tương thích hoàn toàn với thư viện TensorFlow 2.20.0 và TensorFlow Recommenders).
- **Git** cài sẵn trên máy.
- Một dự án **Firebase** đã được khởi tạo và cấu hình cơ sở dữ liệu **Cloud Firestore**.

### 📥 Các bước cài đặt chi tiết

#### Bước 1: Tải mã nguồn về máy
```bash
git clone https://github.com/hoaDaiNhan74/travel-ai-server.git
cd travel-ai-server
```

#### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows (PowerShell)
.\venv\Scripts\activate

# Kích hoạt trên Windows (Command Prompt - CMD)
.\venv\Scripts\activate.bat

# Kích hoạt trên Linux/macOS
source venv/bin/activate
```

#### Bước 3: Cài đặt các thư viện phụ thuộc
Bạn có thể sử dụng công cụ cài đặt thư viện tiêu chuẩn:
```bash
pip install -r requirements.txt
```
*(Nếu máy bạn đã cài sẵn `uv`, có thể dùng lệnh `uv pip install -r requirements.txt` để tăng tốc độ cài đặt gấp 10 lần).*

---

## ⚙️ Cấu Hình Hệ Thống

Trước khi khởi chạy server, bạn cần hoàn tất 3 bước cấu hình bắt buộc sau tại thư mục gốc của dự án:

### 1. File cấu hình môi trường `.env`
Tạo một file có tên chính xác là `.env` ở thư mục gốc và điền thông tin sau:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here
```
- Đăng ký và lấy Groq Key miễn phí tại [Groq Console](https://console.groq.com).
- Đăng ký và lấy Unsplash Key tại [Unsplash Developer Portal](https://unsplash.com/developers).

### 2. Xác thực Firebase Admin SDK (`serviceAccountKey.json`)
Hệ thống cần quyền đọc/ghi Firestore:
1. Mở **Firebase Console** -> Chọn dự án của bạn -> Chọn **Project Settings** (biểu tượng bánh răng).
2. Di chuyển đến tab **Service Accounts**.
3. Nhấp vào **Generate New Private Key** để tải xuống file xác thực JSON.
4. Đổi tên file tải về thành `serviceAccountKey.json` và di chuyển vào thư mục gốc của dự án.

### 3. Khởi tạo mô hình Machine Learning (`two_tower_model`)
Hệ thống cần thư mục `two_tower_model` nằm ở thư mục gốc. Thư mục này chứa mô hình TensorFlow đã được xuất khẩu dưới dạng `SavedModel`. Cấu trúc thư mục phải như sau:
```text
two_tower_model/
├── assets/
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── saved_model.pb
```
*Lưu ý: Nếu không có thư mục này, server vẫn sẽ khởi động bình thường nhưng tính năng gợi ý cá nhân hóa sẽ báo lỗi và tự động kích hoạt chế độ Fallback sang địa điểm thịnh hành.*

---

## 🚀 Cách Khởi Chạy Server

### 1. Chạy thông qua script cấu hình sẵn (Khuyên dùng)
Hệ thống đã viết cấu hình chạy tối ưu hóa sẵn bên trong [main.py](file:///D:/DACN2/triphero_api/main.py):
```bash
python main.py
```

### 2. Chạy thủ công bằng Uvicorn
Nếu bạn muốn tùy biến host hoặc port:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --no-reload
```

> [!WARNING]
> **Không dùng cờ `--reload` khi chạy server!**
> Việc sử dụng tính năng tự động tải lại code của Uvicorn khi phát triển (`--reload`) sẽ khiến toàn bộ mô hình TensorFlow dung lượng lớn bị nạp lại liên tục mỗi khi lưu file. Điều này dẫn đến tràn bộ nhớ RAM (Out of Memory - OOM) và làm sập server ngay lập tức.

---

## 🛠️ Các Script Chạy Thủ Công (Database & Admin Tools)

Thư mục [scripts/](file:///D:/DACN2/triphero_api/scripts) cung cấp các công cụ dòng lệnh quan trạng để quản trị và khởi tạo dữ liệu:

- **Huấn luyện mô hình phân cụm người dùng & xuất biểu đồ**:
  ```bash
  python scripts/train_analytics.py
  ```
  Chạy hàm [run_analytics_pipeline](file:///D:/DACN2/triphero_api/scripts/train_analytics.py#L48-L343) để kéo dữ liệu từ Firestore, chạy thuật toán K-Means phân khúc khách hàng, cập nhật phân khúc ngược lại Firestore, và vẽ 4 biểu đồ lưu vào `uploads/charts/`.

- **Huấn luyện lại mô hình Two-Tower gợi ý thủ công**:
  ```bash
  python scripts/retrain_pipeline.py
  ```
  Hệ thống sẽ kéo dữ liệu tương tác 30 ngày qua trên Firestore, huấn luyện lại mô hình TensorFlow Recommenders, và ghi đè an toàn vào thư mục `two_tower_model/`.

- **Đổ dữ liệu mẫu tương tác người dùng (Interactions Seeding)**:
  ```bash
  python scripts/seed_real_interactions.py
  ```
  Dùng để sinh dữ liệu mẫu các loại tương tác người dùng (view, rate, favorite, share) và tải lên Firestore nhằm phục vụ việc kiểm thử tiến trình huấn luyện gợi ý.

- **Đổ dữ liệu địa điểm ban đầu (Migration)**:
  ```bash
  python scripts/migrate_destinations.py
  ```
  Dùng để đọc file dữ liệu địa điểm du lịch địa phương và đẩy hàng loạt (batch write) lên Firestore làm cơ sở dữ liệu ban đầu.

---

## 🔍 Kiểm Tra & Sử Dụng API

Sau khi khởi động server thành công, truy cập vào các địa chỉ sau:
- **Trang quản lý trạng thái dịch vụ**: `http://localhost:8000/` (Trả về JSON thông tin phiên bản và kiểm tra kết nối TensorFlow, Firebase, Groq).
- **Tài liệu API tương tác (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs) - Nơi bạn có thể trực tiếp test thử các Endpoint (Lập lịch trình, Gợi ý địa điểm, Chatbot, Phân tích hành vi).
- **Endpoint kiểm tra sức khỏe hệ thống (Health Check)**: `http://localhost:8000/health` (Được Flutter client gọi để đánh thức server khi chạy trên Render.com free-tier).

---

## ☁️ Triển Khai Hệ Thống (Deployment)

Server TripHero được cấu hình sẵn để dễ dàng triển khai lên **Render.com** hoặc các dịch vụ tương tự:
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `python main.py` hoặc sử dụng [Procfile](file:///D:/DACN2/triphero_api/Procfile) với Gunicorn:
   ```text
   web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --timeout 120
   ```
3. **Biến môi trường (Environment Variables)**: Cấu hình `GROQ_API_KEY` và `UNSPLASH_ACCESS_KEY` trên dashboard của Render.
4. **Đường dẫn kiểm tra sức khỏe (Health Check Path)**: Thiết lập `/health` để tự động khởi động lại container khi có sự cố.
