import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 경로 설정
    base_path = 'sentiment'
    features_path = os.path.join(base_path, 'features.pkl')
    model_path = os.path.join(base_path, 'model.pkl')

    # 1. 데이터 로드
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found. Please run extract_features.py first.")
        return

    print(f"Loading features from {features_path}...")
    data = joblib.load(features_path)
    X = data['X']
    y = data['y']
    print(f"Loaded feature matrix: {X.shape}")

    # 2. 훈련/테스트 데이터 분할 (7:3)
    print("\nSplitting data: 70% train, 30% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 3. Logistic Regression 모델 학습
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete!")

    # 4. 성능 평가
    y_pred = model.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['부정(0)', '긍정(1)']))

    # 5. 모델 저장
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
