"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TRIPHERO — TWO-TOWER RETRAIN PIPELINE                             ║
║          Phien ban: 2.0.0  |  Dinh dang model: TF SavedModel               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CHUC NANG:                                                                ║
║  1. fetch_data()               — Kéo interactions và destinations (metadata)║
║  2. retrain_two_tower()        — Huấn luyện lại model từ đầu (incorporate   ║
║                                  cả Content-based features).               ║
║  3. Scheduler APScheduler      — Lên lịch tự động chạy 02:00 AM Chủ Nhật    ║
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
from typing import List, Dict, Any, Tuple

# ─── Đảm bảo Python tìm được root package của triphero_api ──────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

# Đảm bảo dùng TF Keras legacy nếu cần cho TFRS (giống train_two_tower.py)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tensorflow_recommenders as tfrs
except ImportError:
    pass

# ─── Logging Setup ───────────────────────────────────────────────────────────────────
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

class RetrainConfig:
    MODEL_DIR          = ROOT_DIR / "two_tower_model"         
    BACKUP_DIR         = ROOT_DIR / "two_tower_model_backups" 

    FIRESTORE_CRED     = ROOT_DIR / "serviceAccountKey.json"
    INTERACTIONS_COL   = "user_interactions"   
    DESTINATIONS_COL   = "destinations"   

    LOOKBACK_DAYS      = 30      
    MIN_INTERACTIONS   = 50     
    INTERACTION_TYPES  = {       
        "view_detail":       1.0,
        "view_map":          1.5,
        "dwell_time":        2.0,
        "share_destination": 3.0,
        "add_favorite":      4.0,
        "rate_destination":  5.0,
        "plan_trip":         5.0,
    }

    EPOCHS             = 15       
    BATCH_SIZE         = 256
    LEARNING_RATE      = 0.1
    EMBEDDING_DIM      = 32

    RETRAIN_DAY        = "sun"   
    RETRAIN_HOUR       = 2       
    RETRAIN_MINUTE     = 0

cfg = RetrainConfig()

def _get_firestore_client():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            if not cfg.FIRESTORE_CRED.exists():
                raise FileNotFoundError(f"Missing {cfg.FIRESTORE_CRED}")
            cred = credentials.Certificate(str(cfg.FIRESTORE_CRED))
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK khoi tao thanh cong.")

        return firestore.client()
    except Exception as e:
        logger.error(f"Khong the ket noi Firestore: {e}")
        return None

def fetch_data(db, lookback_days: int = cfg.LOOKBACK_DAYS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if db is None:
        raise ValueError("Firestore chua ket noi.")

    cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    logger.info(f"Dang fetch data tu {cutoff_time.strftime('%Y-%m-%d')}...")

    # 1. Fetch Destinations
    logger.info("Fetching destinations metadata...")
    dest_docs = db.collection(cfg.DESTINATIONS_COL).stream()
    destinations = []
    dest_dict = {}
    for doc in dest_docs:
        d = doc.to_dict()
        category = d.get("category", "Unknown")
        tags_list = d.get("tags", [])
        if isinstance(tags_list, str):
            tags_list = [tags_list]
        tags = " ".join(tags_list)
        
        dest_item = {
            "destination_id": doc.id,
            "category": category,
            "tags": tags
        }
        destinations.append(dest_item)
        dest_dict[doc.id] = dest_item

    # 2. Fetch Interactions
    logger.info("Fetching interactions...")
    inter_docs = (
        db.collection(cfg.INTERACTIONS_COL)
        .where("timestamp", ">=", cutoff_time)
        .stream()
    )

    interactions = []
    for doc in inter_docs:
        d = doc.to_dict()
        dest_id = d.get("destinationId", "")
        if not dest_id or dest_id == "global" or dest_id not in dest_dict:
            continue

        event_type = d.get("interactionType") or d.get("eventType") or "view_detail"
        rating = cfg.INTERACTION_TYPES.get(event_type, 1.0)
        
        # Merge with destination metadata
        dest_meta = dest_dict[dest_id]

        interactions.append({
            "user_id":        d.get("userId", "unknown"),
            "destination_id": dest_id,
            "category":       dest_meta["category"],
            "tags":           dest_meta["tags"],
            "rating":         rating,
        })

    logger.info(f"Fetch thanh cong: {len(destinations)} destinations, {len(interactions)} interactions hop le.")
    return interactions, destinations

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, cfg.EMBEDDING_DIM)
        ])

    def call(self, inputs):
        return self.user_embedding(inputs)


class DestinationModel(tf.keras.Model):
    def __init__(self, unique_dest_ids, unique_categories):
        super().__init__()
        self.id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_dest_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_dest_ids) + 1, cfg.EMBEDDING_DIM)
        ])
        
        self.category_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_categories, mask_token=None),
            tf.keras.layers.Embedding(len(unique_categories) + 1, cfg.EMBEDDING_DIM)
        ])
        
        self.tags_vectorizer = tf.keras.layers.TextVectorization()
        self.tags_embedding = tf.keras.Sequential([
            self.tags_vectorizer,
            tf.keras.layers.Embedding(1000, cfg.EMBEDDING_DIM, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        
        self.dense_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(cfg.EMBEDDING_DIM)
        ])

    def call(self, inputs):
        return self.dense_net(tf.concat([
            self.id_embedding(inputs["destination_id"]),
            self.category_embedding(inputs["category"]),
            self.tags_embedding(inputs["tags"]),
        ], axis=1))


class TripHeroTwoTowerModel(tfrs.Model):
    def __init__(self, user_model, item_model, candidates_ds):
        super().__init__()
        self.user_tower = user_model
        self.item_tower = item_model
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.batch(128).map(self.item_tower)
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_tower(features["user_id"])
        item_embeddings = self.item_tower({
            "destination_id": features["destination_id"],
            "category": features["category"],
            "tags": features["tags"]
        })
        return self.task(user_embeddings, item_embeddings)


def retrain_two_tower(interactions: List[Dict[str, Any]], destinations: List[Dict[str, Any]]) -> bool:
    try:
        import tensorflow_recommenders as tfrs
    except ImportError:
        logger.error("tensorflow-recommenders is not installed.")
        return False

    if len(interactions) < cfg.MIN_INTERACTIONS:
        logger.warning(f"Chi co {len(interactions)} interactions. Khong du de retrain.")
        return False

    logger.info("Bat dau build dataset...")
    
    unique_user_ids = list(set(r["user_id"] for r in interactions))
    unique_dest_ids = list(set(d["destination_id"] for d in destinations))
    unique_categories = list(set(d["category"] for d in destinations))
    all_tags = [d["tags"] for d in destinations]

    interactions_ds = tf.data.Dataset.from_tensor_slices({
        "user_id": [r["user_id"] for r in interactions],
        "destination_id": [r["destination_id"] for r in interactions],
        "category": [r["category"] for r in interactions],
        "tags": [r["tags"] for r in interactions],
    })

    destinations_ds = tf.data.Dataset.from_tensor_slices({
        "destination_id": [d["destination_id"] for d in destinations],
        "category": [d["category"] for d in destinations],
        "tags": [d["tags"] for d in destinations],
    })

    cached_train = interactions_ds.shuffle(100_000, seed=42).batch(cfg.BATCH_SIZE).cache()

    logger.info("Khoi tao kien truc Two-Tower Model...")
    user_model = UserModel(unique_user_ids)
    item_model = DestinationModel(unique_dest_ids, unique_categories)
    
    item_model.tags_vectorizer.adapt(all_tags)

    model = TripHeroTwoTowerModel(user_model, item_model, destinations_ds)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=cfg.LEARNING_RATE))

    logger.info(f"Bat dau huan luyen {cfg.EPOCHS} epochs...")
    history = model.fit(cached_train, epochs=cfg.EPOCHS, verbose=1)

    final_loss = history.history.get("total_loss", [None])[-1]
    
    logger.info("Dong goi mo hinh (BruteForce Index)...")
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_tower)
    index.index_from_dataset(
        tf.data.Dataset.zip((
            destinations_ds.batch(100).map(lambda x: x["destination_id"]),
            destinations_ds.batch(100).map(item_model)
        ))
    )
    
    _ = index(tf.constant([unique_user_ids[0]]))
    
    _backup_current_model()
    
    try:
        tf.saved_model.save(index, str(cfg.MODEL_DIR))
        _write_retrain_metadata(len(interactions), final_loss)
        logger.info(f"RETRAIN THANH CONG! Model da luu tai {cfg.MODEL_DIR}")
        return True
    except Exception as e:
        logger.error(f"Loi khi luu model: {e}")
        _restore_latest_backup()
        return False

# ==========================================
# UTILS
# ==========================================

def _backup_current_model():
    if not cfg.MODEL_DIR.exists(): return
    cfg.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = cfg.BACKUP_DIR / f"two_tower_{timestamp}"
    try:
        shutil.copytree(str(cfg.MODEL_DIR), str(backup_path))
        _cleanup_old_backups(keep=5)
    except Exception as e:
        logger.warning(f"Khong the backup: {e}")

def _cleanup_old_backups(keep: int = 5):
    if not cfg.BACKUP_DIR.exists(): return
    backups = sorted(cfg.BACKUP_DIR.iterdir(), key=lambda p: p.name)
    for old_backup in backups[:-keep]:
        shutil.rmtree(str(old_backup), ignore_errors=True)

def _restore_latest_backup():
    if not cfg.BACKUP_DIR.exists(): return
    backups = sorted(cfg.BACKUP_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    if not backups: return
    try:
        if cfg.MODEL_DIR.exists():
            shutil.rmtree(str(cfg.MODEL_DIR))
        shutil.copytree(str(backups[0]), str(cfg.MODEL_DIR))
        logger.info("Da khoi phuc model tu backup.")
    except Exception as e:
        logger.error(f"Khong the khoi phuc backup: {e}")

def _write_retrain_metadata(n_samples: int, final_loss: float):
    import json
    meta_path = cfg.MODEL_DIR / "retrain_metadata.json"
    meta = {
        "last_retrain_at":  datetime.now().isoformat(),
        "n_interactions":   n_samples,
        "final_loss":       round(final_loss, 6) if final_loss is not None else None,
        "epochs":           cfg.EPOCHS,
        "learning_rate":    cfg.LEARNING_RATE,
        "lookback_days":    cfg.LOOKBACK_DAYS,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except: pass

def run_retrain_pipeline():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"[PIPELINE START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    (ROOT_DIR / "logs").mkdir(exist_ok=True)
    db = _get_firestore_client()
    try:
        interactions, destinations = fetch_data(db, lookback_days=cfg.LOOKBACK_DAYS)
        if not interactions:
            logger.warning("Khong co du lieu interactions.")
            return
        success = retrain_two_tower(interactions, destinations)
    except Exception as e:
        logger.error(f"Loi he thong: {e}")
        import traceback
        traceback.print_exc()
        success = False

    elapsed = time.time() - start_time
    logger.info(f"[PIPELINE END] Trang thai: {'THANH CONG' if success else 'THAT BAI'} | {elapsed:.1f}s")
    logger.info("=" * 70)

def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")
    scheduler.add_job(
        func=run_retrain_pipeline,
        trigger=CronTrigger(day_of_week=cfg.RETRAIN_DAY, hour=cfg.RETRAIN_HOUR, minute=cfg.RETRAIN_MINUTE),
        id="two_tower_retrain", replace_existing=True, misfire_grace_time=3600,
    )
    logger.info("APScheduler da khoi dong.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

def start_background_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
    scheduler.add_job(
        func=run_retrain_pipeline,
        trigger=CronTrigger(day_of_week=cfg.RETRAIN_DAY, hour=cfg.RETRAIN_HOUR, minute=cfg.RETRAIN_MINUTE),
        id="two_tower_retrain", replace_existing=True, misfire_grace_time=3600,
    )
    scheduler.start()
    return scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    (ROOT_DIR / "logs").mkdir(exist_ok=True)
    if args.dry_run:
        logger.info("[DRY RUN] Fetching data...")
        db = _get_firestore_client()
        if db:
            interactions, destinations = fetch_data(db)
            logger.info(f"[DRY RUN] interactions: {len(interactions)}, destinations: {len(destinations)}")
    elif args.daemon:
        start_scheduler()
    else:
        run_retrain_pipeline()
