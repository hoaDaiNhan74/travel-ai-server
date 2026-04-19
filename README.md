# Hướng dẫn Cài đặt & Chạy Server TripHero (Trekko Backend)

Tài liệu này cung cấp hướng dẫn chi tiết từng bước để thiết lập và khởi chạy hệ thống Unified API của TripHero — một giải pháp Back-end mạnh mẽ kết hợp giữa Công cụ Gợi ý (Two-Tower ML) và Trí tuệ nhân tạo (Groq AI) để hỗ trợ du lịch cá nhân hóa.

---

## 🚀 Giới thiệu Tính năng
*   **Hệ thống Gợi ý Cá nhân hóa (Two-Tower ML)**: Sử dụng mô hình TensorFlow để đề xuất các địa điểm du lịch dựa trên lịch sử và sở thích của người dùng.
*   **AI Trip Planner**: Tự động lập lịch trình du lịch chi tiết bằng mô hình Llama-3 (thông qua Groq API).
*   **Trợ lý ảo (Chatbot)**: Hỗ trợ giải đáp thắc mắc du lịch với khả năng nhớ ngữ cảnh và lịch sử trò chuyện.
*   **Hệ thống Tự động Huấn luyện (Retrain Pipeline)**: Tự động thu thập dữ liệu từ Firestore và cập nhật mô hình ML hàng tuần (2:00 AM Chủ Nhật) mà không cần restart server.

---

## 🛠️ Yêu cầu Hệ thống
Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:
1.  **Python (Phiên bản 3.9 - 3.11)**: Khuyên dùng để đảm bảo tính ổn định tối đa cho TensorFlow và TensorFlow Recommenders.
2.  **Git**: Để tải mã nguồn từ GitHub.
3.  **Tài khoản Firebase**: Để truy cập Firestore.
4.  **API Key**: Groq và Unsplash.

---

## 📥 Các bước Cài đặt Chi tiết

### Bước 1: Tải mã nguồn về máy
Mở Terminal (hoặc PowerShell) và chạy lệnh:
```bash
git clone https://github.com/hoaDaiNhan74/travel-ai-server.git
cd travel-ai-server
```

### Bước 2: Khởi tạo Môi trường ảo (Virtual Environment)
Việc này giúp các thư viện của dự án không bị xung đột với các ứng dụng Python khác trên máy của bạn.
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/macOS)
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
Dự án sử dụng các thư viện quan trọng như `fastapi`, `tensorflow`, `firebase-admin`, `groq`, v.v.
```bash
pip install -r requirements.txt
```

---

## ⚙️ Cấu hình Hệ thống (Quan trọng)

Để server hoạt động đầy đủ tính năng, bạn cần chuẩn bị 3 thành phần sau:

### 1. File biến môi trường (`.env`)
Tạo một file tên là `.env` tại thư mục gốc của dự án với nội dung sau:
```env
GROQ_API_KEY=gsk_xxx... (Lấy tại console.groq.com)
UNSPLASH_ACCESS_KEY=your_key... (Lấy tại Unsplash Developers)
```

### 2. File xác thực Firebase (`serviceAccountKey.json`)
1. Truy cập vào **Firebase Console** -> **Project Settings** -> **Service Accounts**.
2. Nhấn **Generate New Private Key** để tải file JSON về.
3. Đổi tên file vừa tải thành `serviceAccountKey.json` và sao chép vào thư mục gốc của server.

### 3. Thư mục mô hình ML (`two_tower_model`)
Đảm bảo thư mục `two_tower_model/` đã có sẵn tại thư mục gốc. Thư mục này phải chứa file `saved_model.pb` và thư mục `variables/`. Nếu thiếu, tính năng Gợi ý sẽ báo lỗi nhưng các tính năng AI khác vẫn hoạt động.

---

## 🏃 Cách Khởi chạy Server

### Cách 1: Chạy bằng file main.py (Khuyên dùng)
Lệnh này đã được cấu hình sẵn các tham số tối ưu:
```bash
python main.py
```

### Cách 2: Chạy thủ công bằng Uvicorn
Trong trường hợp bạn muốn tùy chỉnh cổng (port) hoặc host:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --no-reload
```
> **Lưu ý**: Tuyệt đối **KHÔNG** dùng `--reload` khi đang chạy với mô hình TensorFlow lớn, vì nó sẽ khiến server nạp lại mô hình liên tục dẫn đến tràn RAM (OOM).

---

## 🔍 Kiểm tra & Sử dụng API

Sau khi server khởi động thành công, bạn có thể kiểm tra qua:
*   **Trang chủ**: `http://localhost:8000/` (Trả về trạng thái các dịch vụ).
*   **Tài liệu API (Swagger)**: `http://localhost:8000/docs` - Đây là nơi bạn có thể dùng thử trực tiếp các endpoint như `/api/v1/recommend`, `/api/v1/trips/generate`, v.v.
*   **Kiểm tra sức khỏe**: `http://localhost:8000/health`

---

## ☁️ Hướng dẫn Triển khai (Deployment)
Nếu bạn muốn đưa server lên các nền tảng đám mây như **Render.com**:
1.  **Build Command**: `pip install -r requirements.txt`
2.  **Start Command**: `python main.py`
3.  **Environment Variables**: Đừng quên thêm các biến trong file `.env` vào phần cấu hình của Render.
4.  **Health Check Path**: Đặt là `/health`.
