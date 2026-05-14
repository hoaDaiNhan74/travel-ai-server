import firebase_admin
from firebase_admin import credentials, firestore
import os
from pathlib import Path

# --- CẤU HÌNH ---
ROOT_DIR = Path(__file__).resolve().parent.parent
CRED_PATH = ROOT_DIR / "serviceAccountKey.json"

def migrate_destinations():
    """
    Script cập nhật các trường số liệu thực tế cho toàn bộ Destinations cũ.
    Giúp Firestore Index hoạt động chính xác khi thực hiện ORDER BY.
    """
    
    # 1. Khởi tạo Firebase
    if not os.path.exists(CRED_PATH):
        print(f"❌ Không tìm thấy file: {CRED_PATH}")
        return

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(CRED_PATH))
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    dest_ref = db.collection("destinations")
    
    print("🔍 Đang quét danh sách địa điểm...")
    docs = dest_ref.stream()
    
    batch = db.batch()
    count = 0
    total_updated = 0

    for doc in docs:
        data = doc.to_dict()
        needs_update = False
        update_payload = {}

        # Kiểm tra và gán giá trị mặc định nếu thiếu
        if "viewCount" not in data:
            update_payload["viewCount"] = 0
            needs_update = True
        
        if "favoriteCount" not in data:
            update_payload["favoriteCount"] = 0
            needs_update = True

        if "ratingCount" not in data:
            update_payload["ratingCount"] = 0
            needs_update = True

        if "averageRating" not in data:
            update_payload["averageRating"] = 0.0
            needs_update = True
        
        # Nếu thiếu rating cũ (trường cũ của bạn), có thể copy sang averageRating
        if "rating" in data and "averageRating" not in data:
            update_payload["averageRating"] = float(data["rating"])
            needs_update = True

        if needs_update:
            batch.update(doc.reference, update_payload)
            count += 1
            total_updated += 1
            
            # Firestore batch giới hạn 500 operations
            if count >= 400:
                batch.commit()
                print(f"✅ Đã cập nhật xong {total_updated} địa điểm...")
                batch = db.batch()
                count = 0

    if count > 0:
        batch.commit()
        print(f"✅ Đã cập nhật xong {total_updated} địa điểm cuối cùng.")

    print("\n" + "="*40)
    print(f"🎉 HOÀN TẤT: Đã đồng bộ hóa {total_updated} địa điểm.")
    print("="*40)

if __name__ == "__main__":
    migrate_destinations()
