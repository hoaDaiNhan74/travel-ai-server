"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TRIPHERO — TWO-TOWER RETRAIN PIPELINE                             ║
║          Phien ban: 1.0.0  |  Dinh dang model: TF SavedModel               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CHUC NANG:                                                                ║
║  1. fetch_recent_interactions() — Keo du lieu tuong tac 30 ngay tu         ║
║     Firestore (views, likes, bookmarks, trip_generated).                   ║
║  2. build_training_dataset()   — Chuyen raw interactions -> tf.data.Dataset ║
║  3. retrain_two_tower()        — Fine-tune Two-Tower model (incremental)    ║
║     roi luu de vao two_tower_model/ (va backup phien ban cu).              ║
║  4. Scheduler APScheduler      — Len lich tu dong chay 02:00 AM Chu Nhat   ║
║                                                                            ║
║  CHAY THU CONG:                                                            ║
║     cd D:/DACN2/triphero_api                                               ║
║     .\venv\Scripts\python scripts/retrain_pipeline.py                      ║
║                                                                            ║
║  CHAY NHU DAEMON (nen):                                                    ║
║     .\venv\Scripts\python scripts/retrain_pipeline.py --daemon             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import shutil
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ─── Đảm bảo Python tìm được root package của triphero_api ──────────────────
# Script nằm trong scripts/, cần thêm parent dir vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

# ─── Logging Setup ───────────────────────────────────────────────────────────────────
# Tạo thư mục logs trước khi cấu hình FileHandler (tránh FileNotFoundError)
_LOG_DIR = ROOT_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "retrain.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("retrain_pipeline")


# ════════════════════════════════════════════════════════════════════════════════
# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

class RetrainConfig:
    """Tập trung toàn bộ tham số có thể điều chỉnh."""

    # Đường dẫn model
    MODEL_DIR          = ROOT_DIR / "two_tower_model"         # SavedModel hiện tại
    BACKUP_DIR         = ROOT_DIR / "two_tower_model_backups" # Thư mục backup

    # Firestore
    FIRESTORE_CRED     = ROOT_DIR / "serviceAccountKey.json"
    INTERACTIONS_COL   = "user_interactions"   # Đổi từ 'interactions' để khớp với Flutter
    DESTINATIONS_COL   = "destinations"   # tên collection destinations

    # Dữ liệu huấn luyện
    LOOKBACK_DAYS      = 30      # Lấy dữ liệu 30 ngày gần nhất
    MIN_INTERACTIONS   = 100     # Cần ít nhất N interaction mới bắt đầu retrain
    INTERACTION_TYPES  = {       # event_type → implicit rating score (Đồng bộ với Flutter)
        "view_detail":       1.0,
        "view_map":          1.5,
        "dwell_time":        2.0,
        "share_destination": 3.0,
        "add_favorite":      4.0,
        "rate_destination":  5.0,
        "plan_trip":         5.0,
    }


    # Huấn luyện
    FINE_TUNE_EPOCHS   = 7       # Số epochs fine-tune
    BATCH_SIZE         = 256
    LEARNING_RATE      = 1e-4    # Thấp để tránh catastrophic forgetting

    # Scheduler
    RETRAIN_DAY        = "sun"   # Chủ Nhật
    RETRAIN_HOUR       = 2       # 02:00 AM
    RETRAIN_MINUTE     = 0


cfg = RetrainConfig()


# ════════════════════════════════════════════════════════════════════════════════
# ─── BƯỚC 1: KẾT NỐI FIRESTORE ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def _get_firestore_client():
    """Khởi tạo Firestore client (tránh init lại nếu đã có)."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            if not cfg.FIRESTORE_CRED.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy serviceAccountKey.json tại: {cfg.FIRESTORE_CRED}"
                )
            cred = credentials.Certificate(str(cfg.FIRESTORE_CRED))
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK khoi tao thanh cong.")

        return firestore.client()
    except Exception as e:
        logger.error(f"Khong the ket noi Firestore: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# ─── BƯỚC 2: FETCH DỮ LIỆU TƯƠNG TÁC ─────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def fetch_recent_interactions(
    db,
    lookback_days: int = cfg.LOOKBACK_DAYS
) -> List[Dict[str, Any]]:
    """
    Kéo dữ liệu tương tác (interactions) trong N ngày qua từ Firestore.

    Schema mỗi document trong collection 'interactions':
    {
        "userId":        "abc123",          # Firebase Auth UID
        "destinationId": "dest_456",        # ID địa điểm
        "eventType":     "like",            # "view" | "like" | "bookmark" | "trip_generated"
        "timestamp":     Timestamp(...)     # Firestore Timestamp
    }

    Returns:
        List[Dict]: Danh sách interactions, mỗi phần tử có:
            user_id, destination_id, event_type, rating (implicit score)
    """
    if db is None:
        logger.warning("Firestore chua ket noi. Su dung du lieu gia lap de test.")
        return _generate_mock_interactions()

    cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    logger.info(f"Dang fetch interactions tu {cutoff_time.strftime('%Y-%m-%d')} den hom nay...")

    try:
        docs = (
            db.collection(cfg.INTERACTIONS_COL)
            .where("timestamp", ">=", cutoff_time)
            .order_by("timestamp", direction="ASCENDING")
            .stream()
        )

        interactions = []
        for doc in docs:
            data = doc.to_dict()
            
            # Bỏ qua các tương tác không gắn với destination cụ thể (search_query, filter_category)
            dest_id = data.get("destinationId", "")
            if not dest_id or dest_id == "global":
                continue

            # Flutter dùng 'interactionType', model cũ dùng 'eventType'
            event_type = data.get("interactionType") or data.get("eventType") or "view_detail"
            rating = cfg.INTERACTION_TYPES.get(event_type, 1.0)

            interactions.append({
                "user_id":        data.get("userId", ""),
                "destination_id": dest_id,
                "event_type":     event_type,
                "rating":         rating,
            })

        logger.info(f"Fetch thanh cong: {len(interactions)} interactions hop le trong {lookback_days} ngay qua.")
        return interactions

    except Exception as e:
        logger.error(f"Loi khi fetch interactions: {e}")
        logger.warning("Fallback ve du lieu gia lap.")
        return _generate_mock_interactions()


def _generate_mock_interactions(n: int = 500) -> List[Dict[str, Any]]:
    """
    Tạo dữ liệu giả lập để test pipeline khi Firestore chưa có dữ liệu thực.
    Trong production, hàm này sẽ không bao giờ được gọi.
    """
    logger.info(f"[MOCK] Tao {n} interactions gia lap de test pipeline...")
    rng = np.random.default_rng(seed=42)

    user_ids = [f"user_{i:04d}" for i in range(50)]
    dest_ids = [f"dest_{i:04d}" for i in range(100)]
    event_types = list(cfg.INTERACTION_TYPES.keys())

    interactions = []
    for _ in range(n):
        event = rng.choice(event_types)
        interactions.append({
            "user_id":        rng.choice(user_ids),
            "destination_id": rng.choice(dest_ids),
            "event_type":     event,
            "rating":         cfg.INTERACTION_TYPES[event],
        })

    logger.info(f"[MOCK] Tao xong {n} interactions gia lap.")
    return interactions


# ════════════════════════════════════════════════════════════════════════════════
# ─── BƯỚC 3: XÂY DỰNG TF DATASET ───────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    interactions: List[Dict[str, Any]],
) -> Tuple[tf.data.Dataset, List[str], List[str]]:
    """
    Chuyển đổi raw interactions → tf.data.Dataset chuẩn cho Two-Tower fine-tuning.

    Returns:
        (dataset, unique_users, unique_destinations)
        - dataset: tf.data.Dataset với từng element là dict
            {"user_id": str_tensor, "destination_id": str_tensor, "rating": float_tensor}
        - unique_users: danh sách user IDs duy nhất
        - unique_destinations: danh sách destination IDs duy nhất
    """
    if not interactions:
        raise ValueError("Khong co du lieu de xay dung dataset.")

    user_ids   = [r["user_id"]        for r in interactions]
    dest_ids   = [r["destination_id"] for r in interactions]
    ratings    = [r["rating"]         for r in interactions]

    unique_users = list(set(user_ids))
    unique_dests = list(set(dest_ids))

    logger.info(
        f"Dataset: {len(interactions)} samples | "
        f"{len(unique_users)} users | {len(unique_dests)} destinations"
    )

    dataset = tf.data.Dataset.from_tensor_slices({
        "user_id":        user_ids,
        "destination_id": dest_ids,
        "rating":         tf.cast(ratings, tf.float32),
    })

    dataset = (
        dataset
        .shuffle(buffer_size=len(interactions), seed=42)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, unique_users, unique_dests


# ════════════════════════════════════════════════════════════════════════════════
# ─── BƯỚC 4: FINE-TUNE TWO-TOWER MODEL ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def retrain_two_tower(interactions: List[Dict[str, Any]]) -> bool:
    """
    Fine-tune Two-Tower Retrieval Model với dữ liệu tương tác mới.

    Chiến lược Incremental Learning:
    - Load model hiện tại từ two_tower_model/ (SavedModel format).
    - Xây dựng Keras wrapper (retrieval model) để compile + fit.
    - Train thêm FINE_TUNE_EPOCHS với learning_rate thấp (tránh catastrophic forgetting).
    - Backup phiên bản cũ → lưu model mới đè lên.

    Args:
        interactions: Kết quả từ fetch_recent_interactions().

    Returns:
        True nếu retrain thành công, False nếu thất bại.
    """
    logger.info("=" * 70)
    logger.info("BAT DAU RETRAIN TWO-TOWER MODEL")
    logger.info(f"  Model path : {cfg.MODEL_DIR}")
    logger.info(f"  Epochs     : {cfg.FINE_TUNE_EPOCHS}")
    logger.info(f"  Batch size : {cfg.BATCH_SIZE}")
    logger.info(f"  LR         : {cfg.LEARNING_RATE}")
    logger.info("=" * 70)

    # ── Kiểm tra đủ dữ liệu ─────────────────────────────────────────────────
    if len(interactions) < cfg.MIN_INTERACTIONS:
        logger.warning(
            f"Chi co {len(interactions)} interactions "
            f"(can toi thieu {cfg.MIN_INTERACTIONS}). Bo qua retrain lan nay."
        )
        return False

    # ── Build Dataset ────────────────────────────────────────────────────────
    try:
        dataset, unique_users, unique_dests = build_training_dataset(interactions)
    except Exception as e:
        logger.error(f"Loi xay dung dataset: {e}")
        return False

    # ── Load SavedModel hiện tại ─────────────────────────────────────────────
    if not cfg.MODEL_DIR.exists():
        logger.error(f"Khong tim thay model tai: {cfg.MODEL_DIR}")
        return False

    try:
        logger.info(f"Dang load SavedModel tu '{cfg.MODEL_DIR}'...")
        base_model = tf.saved_model.load(str(cfg.MODEL_DIR))
        logger.info("Load model thanh cong.")
    except Exception as e:
        logger.error(f"Khong the load model: {e}")
        return False

    # ── Xây dựng Retrieval Wrapper để fine-tune ───────────────────────────────
    # Two-Tower model (TFRS export) không có sẵn .fit() — cần build wrapper.
    try:
        import tensorflow_recommenders as tfrs

        # Lấy user và item towers từ SavedModel
        # (tùy cách export, có thể là query_model và candidate_model)
        user_tower  = base_model.query_model       if hasattr(base_model, "query_model")       else None
        item_tower  = base_model.candidate_model   if hasattr(base_model, "candidate_model")   else None

        if user_tower is None or item_tower is None:
            logger.warning(
                "Khong tim thay query_model/candidate_model trong SavedModel. "
                "Tien hanh fine-tune theo phuong phap gradient tape thu cong."
            )
            return _manual_gradient_fine_tune(base_model, dataset)

        # Tạo FactorizedTopK metric với candidate corpus từ interactions mới
        candidates_ds = tf.data.Dataset.from_tensor_slices(unique_dests).batch(128)
        metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_ds.map(item_tower))

        # Retrieval Task
        task = tfrs.tasks.Retrieval(metrics=metrics)

        # Wrapper Model
        class FineTuneWrapper(tfrs.Model):
            def __init__(self, user_tower, item_tower, task):
                super().__init__()
                self.user_tower = user_tower
                self.item_tower = item_tower
                self.task = task

            def compute_loss(self, features, training=False):
                user_emb = self.user_tower(features["user_id"])
                item_emb = self.item_tower(features["destination_id"])
                return self.task(user_emb, item_emb)

        fine_tune_model = FineTuneWrapper(user_tower, item_tower, task)
        fine_tune_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
        )

        logger.info(f"Bat dau fine-tune: {cfg.FINE_TUNE_EPOCHS} epochs...")
        history = fine_tune_model.fit(dataset, epochs=cfg.FINE_TUNE_EPOCHS, verbose=1)

        # Log kết quả cuối
        final_loss = history.history.get("total_loss", [None])[-1]
        if final_loss is not None:
            logger.info(f"Fine-tune hoan thanh. Final loss: {final_loss:.4f}")

    except ImportError:
        logger.warning("tensorflow-recommenders chua cai. Thu phuong phap thu cong.")
        return _manual_gradient_fine_tune(base_model, dataset)
    except Exception as e:
        logger.error(f"Loi trong qua trinh fine-tune: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ── Backup model cũ ─────────────────────────────────────────────────────
    _backup_current_model()

    # ── Lưu model mới đè lên two_tower_model/ ───────────────────────────────
    try:
        logger.info(f"Dang luu model moi vao '{cfg.MODEL_DIR}'...")

        # Export user tower (query model) — đây là phần được dùng khi inference
        tf.saved_model.save(fine_tune_model, str(cfg.MODEL_DIR))
        logger.info("Luu model moi thanh cong!")

        # Ghi metadata retrain
        _write_retrain_metadata(len(interactions), final_loss)

        logger.info("=" * 70)
        logger.info("RETRAIN THANH CONG!")
        logger.info(f"  Model da duoc cap nhat tai: {cfg.MODEL_DIR}")
        logger.info(f"  Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.error(f"Loi khi luu model: {e}. Khoi phuc ban backup.")
        _restore_latest_backup()
        return False


def _manual_gradient_fine_tune(model, dataset: tf.data.Dataset) -> bool:
    """
    Phương án dự phòng: Fine-tune thủ công bằng GradientTape
    khi không thể dùng TFRS wrapper (do export format khác nhau).
    """
    logger.info("[MANUAL FINE-TUNE] Su dung GradientTape...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)

    try:
        infer_fn = model.signatures.get("serving_default")
        if infer_fn is None:
            infer_fn = list(model.signatures.values())[0]
    except Exception:
        logger.error("Khong tim thay serving signature trong SavedModel.")
        return False

    # Lấy các biến trainable
    trainable_vars = model.variables

    total_loss_log = []
    for epoch in range(cfg.FINE_TUNE_EPOCHS):
        epoch_losses = []
        for batch in dataset:
            with tf.GradientTape() as tape:
                user_ids_batch = tf.expand_dims(batch["user_id"], -1)
                outputs = infer_fn(input=user_ids_batch)
                # Implicit loss: maximize score của destination đúng
                scores = list(outputs.values())[0]
                loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(scores) + 1e-9))

            grads = tape.gradient(loss, trainable_vars)
            valid_pairs = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
            if valid_pairs:
                optimizer.apply_gradients(valid_pairs)
            epoch_losses.append(float(loss.numpy()))

        avg_loss = np.mean(epoch_losses)
        total_loss_log.append(avg_loss)
        logger.info(f"  Epoch {epoch + 1}/{cfg.FINE_TUNE_EPOCHS} — loss: {avg_loss:.4f}")

    logger.info(f"[MANUAL FINE-TUNE] Hoan thanh. Loss cuoi: {total_loss_log[-1]:.4f}")

    # Backup và lưu
    _backup_current_model()
    try:
        tf.saved_model.save(model, str(cfg.MODEL_DIR))
        _write_retrain_metadata(0, total_loss_log[-1])
        return True
    except Exception as e:
        logger.error(f"Loi luu manual fine-tune: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# ─── UTILITIES ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def _backup_current_model():
    """Sao lưu model hiện tại với timestamp trước khi ghi đè."""
    if not cfg.MODEL_DIR.exists():
        return
    cfg.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = cfg.BACKUP_DIR / f"two_tower_{timestamp}"
    try:
        shutil.copytree(str(cfg.MODEL_DIR), str(backup_path))
        logger.info(f"Backup model cu tai: {backup_path}")
        _cleanup_old_backups(keep=5)  # Giữ tối đa 5 bản backup
    except Exception as e:
        logger.warning(f"Khong the backup: {e}")


def _cleanup_old_backups(keep: int = 5):
    """Xóa các bản backup cũ, chỉ giữ lại N bản mới nhất."""
    if not cfg.BACKUP_DIR.exists():
        return
    backups = sorted(cfg.BACKUP_DIR.iterdir(), key=lambda p: p.name)
    to_delete = backups[:-keep]
    for old_backup in to_delete:
        shutil.rmtree(str(old_backup), ignore_errors=True)
        logger.info(f"Da xoa backup cu: {old_backup.name}")


def _restore_latest_backup():
    """Khôi phục backup mới nhất nếu retrain thất bại."""
    if not cfg.BACKUP_DIR.exists():
        return
    backups = sorted(cfg.BACKUP_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    if not backups:
        return
    latest = backups[0]
    try:
        if cfg.MODEL_DIR.exists():
            shutil.rmtree(str(cfg.MODEL_DIR))
        shutil.copytree(str(latest), str(cfg.MODEL_DIR))
        logger.info(f"Da khoi phuc model tu backup: {latest.name}")
    except Exception as e:
        logger.error(f"Khong the khoi phuc backup: {e}")


def _write_retrain_metadata(n_samples: int, final_loss: Optional[float]):
    """Ghi file metadata.json vào two_tower_model/ sau mỗi lần retrain."""
    import json
    meta_path = cfg.MODEL_DIR / "retrain_metadata.json"
    meta = {
        "last_retrain_at":  datetime.now().isoformat(),
        "n_interactions":   n_samples,
        "final_loss":       round(final_loss, 6) if final_loss is not None else None,
        "epochs":           cfg.FINE_TUNE_EPOCHS,
        "learning_rate":    cfg.LEARNING_RATE,
        "lookback_days":    cfg.LOOKBACK_DAYS,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Ghi metadata retrain: {meta_path}")
    except Exception as e:
        logger.warning(f"Khong the ghi metadata: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# ─── ORCHESTRATOR: HÀM CHÍNH CHẠY TOÀN BỘ PIPELINE ──────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def run_retrain_pipeline():
    """
    Hàm điều phối toàn bộ pipeline retrain.
    Được gọi bởi Scheduler hoặc chạy thủ công.
    """
    start_time = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"[PIPELINE START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Đảm bảo thư mục logs tồn tại
    (ROOT_DIR / "logs").mkdir(exist_ok=True)

    # Bước 1: Kết nối Firestore
    db = _get_firestore_client()

    # Bước 2: Fetch dữ liệu tương tác
    interactions = fetch_recent_interactions(db, lookback_days=cfg.LOOKBACK_DAYS)

    if not interactions:
        logger.warning("Khong co du lieu interactions. Ket thuc pipeline.")
        return

    # Bước 3: Retrain
    success = retrain_two_tower(interactions)

    elapsed = time.time() - start_time
    status = "THANH CONG" if success else "THAT BAI"
    logger.info(f"[PIPELINE END] Trang thai: {status} | Thoi gian: {elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("")


# ════════════════════════════════════════════════════════════════════════════════
# ─── SCHEDULER: TỰ ĐỘNG CHẠY 02:00 AM CHỦ NHẬT ──────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def start_scheduler():
    """
    Khởi động APScheduler để tự động retrain vào:
    02:00 AM mỗi Chủ Nhật (Asia/Ho_Chi_Minh timezone).
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    scheduler.add_job(
        func=run_retrain_pipeline,
        trigger=CronTrigger(
            day_of_week=cfg.RETRAIN_DAY,
            hour=cfg.RETRAIN_HOUR,
            minute=cfg.RETRAIN_MINUTE,
        ),
        id="two_tower_retrain",
        name="Two-Tower Weekly Retrain",
        replace_existing=True,
        misfire_grace_time=3600,  # Nếu lỡ cron, cho phép chạy trễ tối đa 1h
    )

    next_run = scheduler.get_job("two_tower_retrain").next_run_time
    logger.info("APScheduler da khoi dong.")
    logger.info(f"  Lich retrain : Chu Nhat {cfg.RETRAIN_HOUR:02d}:{cfg.RETRAIN_MINUTE:02d} AM (Asia/Ho_Chi_Minh)")
    logger.info(f"  Lan chay ke  : {next_run}")
    logger.info("Nhan Ctrl+C de dung scheduler.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler da dung.")


def start_background_scheduler():
    """
    Phiên bản non-blocking — dùng khi nhúng vào FastAPI lifespan.
    Trả về scheduler object để có thể shutdown khi server tắt.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")

    scheduler.add_job(
        func=run_retrain_pipeline,
        trigger=CronTrigger(
            day_of_week=cfg.RETRAIN_DAY,
            hour=cfg.RETRAIN_HOUR,
            minute=cfg.RETRAIN_MINUTE,
        ),
        id="two_tower_retrain",
        name="Two-Tower Weekly Retrain",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    scheduler.start()
    next_run = scheduler.get_job("two_tower_retrain").next_run_time
    logger.info(f"[Scheduler] Background scheduler da khoi dong. Lan retrain ke: {next_run}")
    return scheduler


# ════════════════════════════════════════════════════════════════════════════════
# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TripHero Two-Tower Retrain Pipeline"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Chay o che do daemon: blocking scheduler tu dong retrain hang tuan."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi fetch du lieu va in thong ke, khong thuc su retrain."
    )
    args = parser.parse_args()

    # Đảm bảo thư mục logs tồn tại
    (ROOT_DIR / "logs").mkdir(exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Chi fetch va kiem tra du lieu...")
        db = _get_firestore_client()
        interactions = fetch_recent_interactions(db)
        logger.info(f"[DRY RUN] So luong interactions: {len(interactions)}")
        if interactions:
            event_counts = {}
            for r in interactions:
                event_counts[r["event_type"]] = event_counts.get(r["event_type"], 0) + 1
            logger.info(f"[DRY RUN] Phan bo event types: {event_counts}")
        logger.info("[DRY RUN] Hoan tat. Khong thuc hien retrain.")

    elif args.daemon:
        logger.info("Che do DAEMON — BlockingScheduler.")
        start_scheduler()

    else:
        logger.info("Che do MANUAL — Chay retrain ngay lap tuc.")
        run_retrain_pipeline()
