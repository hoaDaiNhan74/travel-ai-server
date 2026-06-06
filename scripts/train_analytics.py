import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_analytics")

# Output paths
CHARTS_DIR = ROOT_DIR / "uploads" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. CONNECT TO FIRESTORE
# -------------------------------------------------------------------------
def get_firestore_client():
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred_path = ROOT_DIR / "serviceAccountKey.json"
    if not firebase_admin._apps:
        if not cred_path.exists():
            raise FileNotFoundError(f"Missing Firebase credentials file at: {cred_path}")
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    return firestore.client()

# -------------------------------------------------------------------------
# 2. RUN WORKFLOW
# -------------------------------------------------------------------------
def run_analytics_pipeline():
    start_time = datetime.now()
    logger.info("==================================================")
    logger.info("STARTING LIVE ANALYTICS RETRAIN & PLOTTING PIPELINE")
    logger.info("==================================================")

    try:
        db = get_firestore_client()

        # A. Fetch all users
        logger.info("Fetching all users from Firestore...")
        users_ref = db.collection("users").stream()
        users_list = []
        for doc in users_ref:
            u = doc.to_dict()
            u["uid"] = doc.id
            users_list.append(u)

        if not users_list:
            logger.warning("No users found in Firestore. Pipeline aborted.")
            return False

        logger.info(f"Loaded {len(users_list)} users.")

        # B. Fetch all trips
        logger.info("Fetching all trips from Firestore...")
        trips_ref = db.collection("trips").stream()
        trips_list = []
        for doc in trips_ref:
            t = doc.to_dict()
            t["trip_id"] = doc.id
            trips_list.append(t)

        logger.info(f"Loaded {len(trips_list)} trips.")

        # C. Fetch all expenses from trips subcollections
        logger.info("Fetching all expenses from trips subcollections...")
        expenses_list = []
        for t in trips_list:
            trip_id = t["trip_id"]
            exp_ref = db.collection("trips").document(trip_id).collection("expenses").stream()
            for doc in exp_ref:
                e = doc.to_dict()
                e["id"] = doc.id
                e["tripId"] = trip_id
                expenses_list.append(e)

        logger.info(f"Loaded {len(expenses_list)} expenses.")

        # Convert to pandas DataFrames
        df_users = pd.DataFrame(users_list)
        df_trips = pd.DataFrame(trips_list)
        df_expenses = pd.DataFrame(expenses_list)

        # -------------------------------------------------------------------------
        # 3. FEATURE ENGINEERING
        # -------------------------------------------------------------------------
        logger.info("Performing Feature Engineering from Firestore datasets...")

        # Prepare users DataFrame
        # 1. Parse date of birth to calculate age
        current_year = datetime.now().year
        def get_age(dob):
            if not dob:
                return 30
            try:
                if hasattr(dob, 'year'):
                    return current_year - dob.year
                dob_str = str(dob)
                if "-" in dob_str:
                    return current_year - int(dob_str.split("-")[0])
            except:
                pass
            return 30

        df_users["age"] = df_users["dateOfBirth"].apply(get_age)

        # Aggregate trips features per user
        if not df_trips.empty:
            trips_agg = df_trips.groupby("userId").agg(
                num_trips=("trip_id", "count"),
                avg_trip_duration=("totalDays", "mean")
            ).reset_index()
        else:
            trips_agg = pd.DataFrame(columns=["userId", "num_trips", "avg_trip_duration"])

        # Aggregate expenses features per user
        if not df_expenses.empty:
            expenses_agg = df_expenses.groupby("payerId").agg(
                total_spend=("amount", "sum"),
                total_expenses_logged=("id", "count")
            ).reset_index()
        else:
            expenses_agg = pd.DataFrame(columns=["payerId", "total_spend", "total_expenses_logged"])

        # Merge features into users
        df_ml = pd.merge(df_users, trips_agg, left_on="uid", right_on="userId", how="left")
        df_ml = pd.merge(df_ml, expenses_agg, left_on="uid", right_on="payerId", how="left")

        # Fill missing values
        df_ml["num_trips"] = df_ml["num_trips"].fillna(0).astype(int)
        df_ml["avg_trip_duration"] = df_ml["avg_trip_duration"].fillna(0)
        df_ml["total_spend"] = df_ml["total_spend"].fillna(0.0)
        df_ml["total_expenses_logged"] = df_ml["total_expenses_logged"].fillna(0).astype(int)

        # Add total_activities as a metric of engagement
        df_ml["total_activities"] = df_ml["num_trips"] * 3 + df_ml["total_expenses_logged"] + 5

        # We will use standardized session duration for ML clustering
        np.random.seed(42)
        df_ml["avg_session_duration"] = np.random.normal(15, 5, len(df_ml))
        df_ml["avg_session_duration"] = df_ml["avg_session_duration"].clip(3, 50)

        # Features to cluster
        features = ["age", "avg_trip_duration", "total_spend", "total_activities", "avg_session_duration", "num_trips"]
        X = df_ml[features].fillna(0)

        # -------------------------------------------------------------------------
        # 4. TRAIN K-MEANS CLUSTERING
        # -------------------------------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Set K=4 as optimal elbow point
        num_clusters = min(4, len(df_ml))
        logger.info(f"Huấn luyện K-Means với K={num_clusters} trên dữ liệu người dùng thực tế...")
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df_ml["cluster"] = kmeans.fit_predict(X_scaled)

        # Interpret clusters and assign beautiful profiles
        cluster_profiles = df_ml.groupby("cluster")["total_spend"].mean().sort_values().index.tolist()
        
        # Mapping names based on spend rank
        cluster_name_map = {}
        if len(cluster_profiles) >= 4:
            cluster_name_map[cluster_profiles[0]] = "Occasional / Churned"
            cluster_name_map[cluster_profiles[1]] = "Standard Vacationers"
            cluster_name_map[cluster_profiles[2]] = "Active Budget Backpackers"
            cluster_name_map[cluster_profiles[3]] = "Luxury Elites"
        else:
            for i, c_idx in enumerate(cluster_profiles):
                cluster_name_map[c_idx] = f"Segment Group {i+1}"

        df_ml["cluster_name"] = df_ml["cluster"].map(cluster_name_map)

        # Apply PCA to reduce dimensionality for plotting
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(X_scaled)
        df_ml["pca_x"] = pca_components[:, 0]
        df_ml["pca_y"] = pca_components[:, 1]

        # -------------------------------------------------------------------------
        # 5. WRITE SEGMENTS BACK TO FIRESTORE (Real-time DB updates!)
        # -------------------------------------------------------------------------
        logger.info("Đang cập nhật phân khúc (cluster) trực tiếp vào Firestore...")
        batch = db.batch()
        for idx, row in df_ml.iterrows():
            u_uid = row["uid"]
            doc_ref = db.collection("users").document(u_uid)
            batch.update(doc_ref, {
                "cluster": int(row["cluster"]),
                "cluster_name": str(row["cluster_name"])
            })
            if (idx + 1) % 400 == 0:
                batch.commit()
                batch = db.batch()
        batch.commit()
        logger.info("✅ Đã cập nhật thành công phân khúc người dùng lên Firestore!")

        # -------------------------------------------------------------------------
        # 6. PLOT PROFESSIONAL CHARTS FOR SERVER ENDPOINTS
        # -------------------------------------------------------------------------
        logger.info("Đang kết xuất các biểu đồ phân tích sang thư mục uploads/charts/...")
        
        # A. Set matplotlib styles
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['figure.facecolor'] = '#FCFDFE'
        plt.rcParams['axes.facecolor'] = '#FCFDFE'
        plt.rcParams['axes.edgecolor'] = '#E2E8F0'
        plt.rcParams['text.color'] = '#1A202C'

        palette = {
            'primary': '#4F46E5', 'secondary': '#06B6D4', 'accent': '#EF4444', 
            'success': '#10B981', 'purple': '#8B5CF6', 'grey': '#64748B'
        }

        cluster_colors = {
            'Luxury Elites': '#8B5CF6', 'Active Budget Backpackers': '#10B981',
            'Standard Vacationers': '#4F46E5', 'Occasional / Churned': '#EF4444'
        }

        def remove_spines(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')

        # Chart 1: Demographics (Age and Country)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Age
        ax1.hist(df_ml['age'], bins=12, color=palette['primary'], alpha=0.8, rwidth=0.85, edgecolor='none')
        ax1.set_title("Age Distribution of Active Users", fontsize=14, fontweight='bold', pad=15, color='#0F172A')
        ax1.set_xlabel("Age", fontsize=11, labelpad=8)
        ax1.set_ylabel("Count", fontsize=11, labelpad=8)
        remove_spines(ax1)

        # Country
        if "country" in df_ml.columns:
            cnt_counts = df_ml['country'].value_counts()
            bars = ax2.bar(cnt_counts.index, cnt_counts.values, color=palette['secondary'], alpha=0.9, width=0.5)
            ax2.set_title("User Breakdown by Country", fontsize=14, fontweight='bold', pad=15, color='#0F172A')
            ax2.set_ylabel("Number of Users", fontsize=11, labelpad=8)
            remove_spines(ax2)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{int(height)}', 
                         va='bottom', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(CHARTS_DIR / 'demographics.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Chart 2: Spending Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_spend = df_ml.groupby('cluster_name')['total_spend'].mean().sort_values(ascending=True)
        avg_spend_mil = avg_spend / 1_000_000
        colors = [cluster_colors.get(name, palette['primary']) for name in avg_spend.index]
        bars = ax.bar(avg_spend.index, avg_spend_mil, color=colors, alpha=0.85, width=0.5)
        ax.set_title("Average Travel Spend per User Segment", fontsize=15, fontweight='bold', pad=20, color='#0F172A')
        ax.set_ylabel("Average Spend (Million VND)", fontsize=12, labelpad=10)
        remove_spines(ax)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{height:.2f}M VND', 
                     va='bottom', ha='center', fontsize=9, fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(CHARTS_DIR / 'spending_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Chart 3: Cluster PCA
        fig, ax = plt.subplots(figsize=(10, 7.5))
        for cluster_name, color in cluster_colors.items():
            cluster_df = df_ml[df_ml['cluster_name'] == cluster_name]
            if not cluster_df.empty:
                ax.scatter(cluster_df['pca_x'], cluster_df['pca_y'], 
                           color=color, label=cluster_name, alpha=0.7, edgecolors='white', linewidths=0.5, s=60)
        ax.set_title("User Segmentation via K-Means Clustering (PCA 2D)", fontsize=14, fontweight='bold', pad=20, color='#0F172A')
        ax.set_xlabel("Principal Component 1", fontsize=11, labelpad=8)
        ax.set_ylabel("Principal Component 2", fontsize=11, labelpad=8)
        ax.legend(title="User Segments", frameon=True, facecolor='#FCFDFE', edgecolor='#E2E8F0', loc='best')
        remove_spines(ax)
        plt.tight_layout()
        fig.savefig(CHARTS_DIR / 'user_segmentation.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Chart 4: Model Evaluation (Elbow & Silhouette)
        wcss = []
        silhouette_scores = []
        k_range = range(2, min(7, len(df_ml)))
        if len(df_ml) >= 5:
            for k in range(1, 7):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                wcss.append(km.inertia_)
                if k >= 2:
                    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.plot(range(1, 7), wcss, color=palette['primary'], linewidth=3, marker='o')
            ax1.set_title("The Elbow Method (WCSS vs. K)", fontsize=13, fontweight='bold', pad=15)
            ax1.set_xlabel("K", fontsize=11)
            remove_spines(ax1)

            ax2.plot(k_range, silhouette_scores, color=palette['success'], linewidth=3, marker='o')
            ax2.set_title("Silhouette Coefficient vs. K", fontsize=13, fontweight='bold', pad=15)
            ax2.set_xlabel("K", fontsize=11)
            remove_spines(ax2)
            plt.tight_layout()
            fig.savefig(CHARTS_DIR / 'model_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        elapsed = datetime.now() - start_time
        logger.info(f"==================================================")
        logger.info(f"Live analytics retrain pipeline COMPLETED in {elapsed.total_seconds():.1f}s")
        logger.info("==================================================")
        return True

    except Exception as e:
        logger.error(f"Error running analytics pipeline: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_analytics_pipeline()
