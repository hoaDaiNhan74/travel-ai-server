import firebase_admin
from firebase_admin import credentials, firestore, auth
import random
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

# --- CẤU HÌNH ---
ROOT_DIR = Path(__file__).resolve().parent.parent
CRED_PATH = ROOT_DIR / "serviceAccountKey.json"

# Cá tính của 10 người dùng
USER_PERSONAS = [
    {"name": "Trần Thanh Tâm", "pref": "Biển", "email": "vanhoahl1@gmail.com"},
    {"name": "Lê Minh Hoàng", "pref": "Núi", "email": "vanhoahl2@gmail.com"},
    {"name": "Nguyễn Thị Mai", "pref": "Văn hóa", "email": "vanhoahl3@gmail.com"},
    {"name": "Phạm Gia Bảo", "pref": "Ẩm thực", "email": "vanhoahl4@gmail.com"},
    {"name": "Hoàng Nhật Anh", "pref": "Biển", "email": "vanhoahl5@gmail.com"},
    {"name": "Vũ Đức Huy", "pref": "Mạo hiểm", "email": "vanhoahl6@gmail.com"},
    {"name": "Đặng Ngọc Diệp", "pref": "Nghỉ dưỡng", "email": "vanhoahl7@gmail.com"},
    {"name": "Bùi Xuân Trường", "pref": "Núi", "email": "vanhoahl8@gmail.com"},
    {"name": "Ngô Bảo Châu", "pref": "Văn hóa", "email": "vanhoahl9@gmail.com"},
    {"name": "Đỗ Hải Đăng", "pref": "Mạo hiểm", "email": "vanhoahl10@gmail.com"},
]
PASSWORD_DEFAULT = "123456"

INTERACTION_SCORES = {
    "view_detail": 1.0,
    "view_map": 1.5,
    "dwell_time": 2.0,
    "share_destination": 3.0,
    "add_favorite": 4.0,
    "rate_destination": 5.0,
    "plan_trip": 5.0,
}

def seed_data():
    if not os.path.exists(CRED_PATH):
        print(f"❌ Không tìm thấy file: {CRED_PATH}")
        return

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(CRED_PATH))
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    print("🚀 Bắt đầu quá trình Seeding dữ liệu người dùng thật...")
    
    # 1. Lấy danh sách destinations để gán tương tác
    destinations = list(db.collection("destinations").stream())
    if not destinations:
        print("❌ Không có địa điểm nào trong database để tạo tương tác.")
        return
    
    dest_list = [d.to_dict() | {"id": d.id} for d in destinations]
    
    for persona in USER_PERSONAS:
        # a. Tạo User trong Firebase Auth (Nếu chưa có)
        try:
            user = auth.get_user_by_email(persona["email"])
            uid = user.uid
            print(f"👤 User đã tồn tại: {persona['name']} ({uid})")
        except firebase_admin.auth.UserNotFoundError:
            user = auth.create_user(
                email=persona["email"],
                password=PASSWORD_DEFAULT,
                display_name=persona["name"]
            )
            uid = user.uid
            # Tạo profile trong Firestore
            db.collection("users").document(uid).set({
                "uid": uid,
                "email": persona["email"],
                "name": persona["name"],
                "bio": f"Tôi là một người yêu thích du lịch {persona['pref']}",
                "createdAt": firestore.SERVER_TIMESTAMP
            })
            print(f"🆕 Đã tạo User mới: {persona['name']} ({uid})")

        # b. Giả lập hành vi tương tác dựa trên sở thích
        # Chọn 5-8 địa điểm ngẫu nhiên, nhưng ưu tiên địa điểm đúng sở thích (pref)
        relevant_dests = [d for d in dest_list if persona["pref"].lower() in str(d.get("category", "")).lower() or persona["pref"].lower() in str(d.get("tags", "")).lower()]
        other_dests = [d for d in dest_list if d not in relevant_dests]
        
        # Mix: 70% sở thích, 30% ngẫu nhiên
        target_dests = random.sample(relevant_dests, min(len(relevant_dests), 5)) + \
                       random.sample(other_dests, min(len(other_dests), 3))
        
        batch = db.batch()
        for dest in target_dests:
            # Mỗi địa điểm sẽ có chuỗi hành động: view -> map -> (tùy chọn) like/rate
            actions = ["view_detail"]
            if random.random() > 0.3: actions.append("view_map")
            if random.random() > 0.5: actions.append("add_favorite")
            if random.random() > 0.7: actions.append("rate_destination")
            if random.random() > 0.8: actions.append("plan_trip")
            
            for action in actions:
                inter_ref = db.collection("user_interactions").document()
                batch.set(inter_ref, {
                    "userId": uid,
                    "destinationId": dest["id"],
                    "interactionType": action,
                    "category": dest.get("category", "General"),
                    "score": INTERACTION_SCORES[action],
                    "timestamp": datetime.now(timezone.utc) - timedelta(days=random.randint(0, 10)),
                    "rating": 5.0 if action == "rate_destination" else None
                })
                
                # Cập nhật số liệu vào destination (increment)
                d_ref = db.collection("destinations").document(dest["id"])
                if action == "view_detail":
                    batch.update(d_ref, {"viewCount": firestore.Increment(1)})
                elif action == "add_favorite":
                    batch.update(d_ref, {"favoriteCount": firestore.Increment(1)})
                elif action == "rate_destination":
                    batch.update(d_ref, {"ratingCount": firestore.Increment(1)})

        batch.commit()
        print(f"   ✅ Đã tạo tương tác cho {persona['name']}")

    print("\n" + "="*50)
    print("🎉 HOÀN TẤT SEEDING! Bạn đã có 10 user với hành vi thực tế.")
    print("Bạn có thể dùng email của họ và mật khẩu 'password123' để login.")
    print("="*50)

if __name__ == "__main__":
    seed_data()
