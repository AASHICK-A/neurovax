import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

SR = 16000
DURATION = 4.0

def extract_robust_features(file_path):
    try:
        try:
            y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        except Exception:
            data, orig_sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            y = librosa.resample(data, orig_sr=orig_sr, target_sr=SR)
            y = y[: int(SR * DURATION)]
            sr = SR

        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) >= int(sr * 0.5):
            y = y_trimmed

        # Peak normalization
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Exact 158 Features: 40 MFCC + 40 Delta + 40 Delta2 + 14 Contrast + 24 Chroma
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        d_mfcc = librosa.feature.delta(mfcc)
        dd_mfcc = librosa.feature.delta(mfcc, order=2)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        feat_list = [
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),        # 40
            np.mean(d_mfcc, axis=1), np.std(d_mfcc, axis=1),    # 40
            np.mean(dd_mfcc, axis=1), np.std(dd_mfcc, axis=1),  # 40
            np.mean(contrast, axis=1), np.std(contrast, axis=1),# 14
            np.mean(chroma, axis=1), np.std(chroma, axis=1)     # 24
        ]
        return np.hstack(feat_list)
    except Exception:
        return None

def main():
    print("=" * 60)
    print("NEURALVOX // FORENSIC RANDOM FOREST TRAINER (158 FEATURES)")
    print("=" * 60)

    human_files = glob.glob("dataset/human/*.wav") + glob.glob("dataset/human/*.mp3")
    ai_files = glob.glob("dataset/ai/*.wav") + glob.glob("dataset/ai/*.mp3")

    print(f"Total Human Clips: {len(human_files)}")
    print(f"Total AI Clips:    {len(ai_files)}")

    if not human_files or not ai_files:
        print("Dataset missing in dataset/human or dataset/ai")
        return

    sample_limit = min(len(human_files), len(ai_files), 1000)
    selected_human = human_files[:sample_limit]
    selected_ai = ai_files[:sample_limit]

    X, y = [], []
    print(f"\nProcessing {sample_limit} Human audio samples...")
    for f in selected_human:
        vec = extract_robust_features(f)
        if vec is not None:
            X.append(vec)
            y.append(0)

    print(f"Processing {sample_limit} AI synthetic samples...")
    for f in selected_ai:
        vec = extract_robust_features(f)
        if vec is not None:
            X.append(vec)
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_depth=28,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100.0

    print(f"\nMODEL ACCURACY: {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"]))

    payload = {
        "model": clf,
        "scaler": scaler,
        "accuracy": acc
    }
    joblib.dump(payload, "voice_model.pkl")
    print("Successfully saved model and scaler to voice_model.pkl")

if __name__ == "__main__":
    main()