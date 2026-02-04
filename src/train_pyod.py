import pandas as pd
import numpy as np
import glob
import pickle
import os
from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
SEQ_LEN = 30  # 30프레임(약 1~2초)을 하나의 패턴으로 봄
MODEL_SAVE_PATH = "data/models/pyod_model.pkl"
SCALER_SAVE_PATH = "data/models/pyod_scaler.pkl"

def create_sequences(data, seq_len):
    """
    (N, Features) 데이터를 (N-seq_len, seq_len * Features)로 변환
    즉, 30개 프레임을 한 줄로 쫙 펴서 하나의 데이터 포인트로 만듦.
    """
    xs = []
    for i in range(len(data) - seq_len):
        x = data[i : i + seq_len]
        # Flatten: (30, 2) -> (60,)
        xs.append(x.flatten())
    return np.array(xs)

def main():
    # 1. Load Data
    csv_files = glob.glob("data/features/*.csv")
    if not csv_files:
        print("❌ 저장된 데이터가 없습니다! src/01_collect_features.py를 실행해서 데이터를 먼저 모으세요.")
        return

    print(f"📂 Found {len(csv_files)} csv files.")
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        df_list.append(df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 필요한 컬럼만 추출 (EAR, Pitch)
    # NaN 값 제거 (얼굴 못 찾았을 때)
    clean_df = full_df[['ear', 'pitch']].dropna()
    print(f"📊 Total Data Points: {len(clean_df)}")

    if len(clean_df) < SEQ_LEN + 10:
        print("❌ 데이터가 너무 적습니다. 더 수집해주세요.")
        return

    # 2. Preprocessing (Scaling)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df.values)
    
    # 3. Create Sequences (Sliding Window)
    # 시계열 특성을 반영하기 위해 윈도우 단위로 데이터를 재가공
    X_train = create_sequences(scaled_data, SEQ_LEN)
    print(f"🧩 Training Shape (Windowed): {X_train.shape}") # (Samples, 60)

    # 4. Train PyOD Model
    # KNN: 가장 직관적 (정상 데이터들과 거리가 멀면 이상)
    # method='mean': 가장 가까운 5개 정상 데이터와의 평균 거리 사용
    print("🤖 Training PyOD KNN model...")
    clf = KNN(method='mean', n_neighbors=5)
    
    # 만약 더 빠른 속도를 원하면 ECOD 사용 가능 (주석 해제)
    # clf = ECOD() 
    
    clf.fit(X_train)
    
    # 5. Save Model
    if not os.path.exists("data/models"):
        os.makedirs("data/models")
        
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(clf, f)
        
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    # Threshold(임계값) 정보 확인
    print(f"✅ Model Saved to {MODEL_SAVE_PATH}")
    print(f"🎯 Threshold (Cut-off Score): {clf.threshold_:.4f}")
    print("이제 src/live_pyod.py를 실행하세요!")

if __name__ == "__main__":
    main()
