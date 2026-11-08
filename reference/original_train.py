import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ----------------------------------------------------
# 0. 환경 설정 및 재현성을 위한 시드 고정
# ----------------------------------------------------
RANDOM_SEED = 0
tf.keras.utils.set_random_seed(RANDOM_SEED)
tf.config.experimental.enable_op_determinism()

# 결과 이미지 저장용 디렉토리 생성
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================
# 1. LFW 데이터셋 로드 및 불균형 해소 (Balancing)
# ====================================================
print("=" * 60)
print("[Step 1] LFW 데이터셋 로드 및 클래스 밸런싱")
print("=" * 60)

lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

n_samples, h, w = lfw_people.images.shape
X_raw = lfw_people.data
images_raw = lfw_people.images
y_raw = lfw_people.target
target_names = lfw_people.target_names
n_available_classes = len(target_names)

print(f"- 원본 데이터셋 크기: {images_raw.shape}, 클래스(인물) 수: {n_available_classes}")

# 전체 62개 클래스를 사용하고, 클래스마다 20장씩 샘플링
required_classes = 62
if n_available_classes != required_classes:
    raise ValueError(
        f"과제에는 {required_classes}개 클래스가 필요하지만 "
        f"데이터셋에서는 {n_available_classes}개 클래스가 확인되었습니다."
    )

rng = np.random.RandomState(RANDOM_SEED)
selected_labels = np.arange(n_available_classes)
selected_target_names = target_names[selected_labels]

balanced_indices = []
balanced_labels = []
for balanced_label, source_label in enumerate(selected_labels):
    label_indices = np.where(y_raw == source_label)[0]
    selected = rng.choice(label_indices, size=20, replace=False)
    balanced_indices.extend(selected)
    balanced_labels.extend([balanced_label] * 20)

balanced_indices = np.array(balanced_indices)
X_balanced = X_raw[balanced_indices]
images_balanced = images_raw[balanced_indices]
y_balanced = np.array(balanced_labels)
n_classes = len(selected_target_names)

print(
    f"- 선택 및 밸런싱 후 데이터셋 크기: {images_balanced.shape} "
    f"({n_classes}명 x 20장 = {len(y_balanced)}장)"
)

# [Figure 1] 밸런싱 전과 후 인물당 이미지 수
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
counts_before = [np.sum(y_raw == label) for label in selected_labels]
plt.bar(range(n_classes), counts_before, color='steelblue')
plt.title("Selected Classes Before Balancing", fontsize=12)
plt.xlabel("Selected Person Class Index")
plt.ylabel("Image Count")

plt.subplot(1, 2, 2)
counts_after = [np.sum(y_balanced == i) for i in range(n_classes)]
plt.bar(range(n_classes), counts_after, color='coral')
plt.title("Number of Images per Person (After Balancing)", fontsize=12)
plt.xlabel("Selected Person Class Index")
plt.ylabel("Image Count")
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "fig1_data_balancing.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"  -> Figure 1 저장 완료: {fig1_path}")


# ====================================================
# 2. Train / Test 세트 분할 (75% / 25%, stratify=y)
# ====================================================
print("\n" + "=" * 60)
print("[Step 2] Train / Test 데이터셋 분할 (75% / 25%, stratify=y)")
print("=" * 60)

# fetch_lfw_people은 픽셀을 [0, 1]로 반환하지만, 범위가 다른 버전에도 대응
pixel_scale = 255.0 if np.max(images_balanced) > 1.0 else 1.0
X_norm = X_balanced / pixel_scale
images_norm = images_balanced / pixel_scale
print(
    f"- CNN/PCA 입력 픽셀 범위: "
    f"[{images_norm.min():.4f}, {images_norm.max():.4f}]"
)

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X_norm, y_balanced, images_norm,
    test_size=0.25,
    stratify=y_balanced,
    random_state=RANDOM_SEED
)

print(f"- Train 세트 크기: {X_train.shape} ({len(y_train)}장)")
print(f"- Test 세트 크기 : {X_test.shape} ({len(y_test)}장)")


# ====================================================
# 3. PCA 화이트닝 분석 및 시각화
# ====================================================
print("\n" + "=" * 60)
print("[Step 3] PCA 화이트닝 분석, 주성분 시각화 및 재구성")
print("=" * 60)

# PCA fit은 반드시 Train 세트에만 수행 (Data Leakage 방지)
pca_no_whiten = PCA(n_components=100, whiten=False, random_state=RANDOM_SEED).fit(X_train)
pca_whiten = PCA(n_components=100, whiten=True, random_state=RANDOM_SEED).fit(X_train)

X_train_pca_nw = pca_no_whiten.transform(X_train)
X_train_pca_w = pca_whiten.transform(X_train)

# [Figure 2] 화이트닝 적용 전과 후의 첫 두 주성분 산점도
class_cmap = plt.get_cmap("turbo", n_classes)
class_norm = plt.Normalize(vmin=-0.5, vmax=n_classes - 0.5)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

scatter1 = axes[0].scatter(
    X_train_pca_nw[:, 0], X_train_pca_nw[:, 1],
    c=y_train, cmap=class_cmap, norm=class_norm, alpha=0.7, s=20
)
axes[0].set_title("First 2 Principal Components (Without Whitening)", fontsize=12)
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")
axes[0].grid(True, linestyle='--', alpha=0.5)

scatter2 = axes[1].scatter(
    X_train_pca_w[:, 0], X_train_pca_w[:, 1],
    c=y_train, cmap=class_cmap, norm=class_norm, alpha=0.7, s=20
)
axes[1].set_title("First 2 Principal Components (With Whitening)", fontsize=12)
axes[1].set_xlabel("PC 1 (Unit Variance)")
axes[1].set_ylabel("PC 2 (Unit Variance)")
axes[1].grid(True, linestyle='--', alpha=0.5)

class_ticks = list(range(0, n_classes - 10, 10)) + [n_classes - 1]
colorbar = fig.colorbar(scatter2, ax=axes, ticks=class_ticks, fraction=0.035, pad=0.03)
colorbar.set_label("Person Class Label")
fig2_path = os.path.join(OUTPUT_DIR, "fig2_pca_whitening_scatter.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
print(f"  -> Figure 2 저장 완료: {fig2_path}")

# [Figure 3] 이미지 형태로 시각화한 주성분 (Eigenfaces)
eigenfaces = pca_no_whiten.components_.reshape((100, h, w))
plt.figure(figsize=(10, 4.5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenfaces[i], cmap='bone')
    plt.title(f"Eigenface {i + 1}", fontsize=11)
    plt.axis('off')
plt.suptitle("Principal Components (Eigenfaces) Visualized as Images", fontsize=13)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "fig3_eigenfaces.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"  -> Figure 3 저장 완료: {fig3_path}")

# [Figure 4] 주성분 개수를 변화시키며 재구성(Reconstruction)한 얼굴 이미지
sample_idx = 0
sample_original = X_train[sample_idx]
component_steps = [10, 30, 50, 100]

plt.figure(figsize=(13, 3))
plt.subplot(1, len(component_steps) + 1, 1)
plt.imshow(sample_original.reshape(h, w), cmap='bone')
plt.title("Original Face", fontsize=11)
plt.axis('off')

for idx, n_comp in enumerate(component_steps):
    temp_pca = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit(X_train)
    projected = temp_pca.transform([sample_original])
    reconstructed = temp_pca.inverse_transform(projected)
    
    plt.subplot(1, len(component_steps) + 1, idx + 2)
    plt.imshow(reconstructed.reshape(h, w), cmap='bone')
    plt.title(f"{n_comp} Components", fontsize=11)
    plt.axis('off')

plt.suptitle("Face Reconstruction with Different Numbers of Principal Components", fontsize=13)
plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "fig4_face_reconstructions.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"  -> Figure 4 저장 완료: {fig4_path}")

# [Figure 5] 누적 설명 분산 플롯 및 90%, 95% 기준 주성분 수 계산
# 학습 샘플 수보다 큰 주성분 수는 만들 수 없으므로 학습 데이터의 중심화된 rank까지 계산
max_variance_components = min(X_train.shape[0] - 1, X_train.shape[1])
pca_variance = PCA(
    n_components=max_variance_components,
    svd_solver='randomized',
    random_state=RANDOM_SEED
).fit(X_train)
cumulative_variance = np.cumsum(pca_variance.explained_variance_ratio_)

def components_for_threshold(cumulative_values, threshold):
    reached = np.flatnonzero(cumulative_values >= threshold)
    return int(reached[0] + 1) if reached.size else None

n_components_90 = components_for_threshold(cumulative_variance, 0.90)
n_components_95 = components_for_threshold(cumulative_variance, 0.95)
variance_90_text = f'{n_components_90}개' if n_components_90 is not None else f'미도달 (최대 {max_variance_components}개 확인)'
variance_95_text = f'{n_components_95}개' if n_components_95 is not None else f'미도달 (최대 {max_variance_components}개 확인)'
print(f"- 분산의 90%를 설명하는 주성분 개수: {variance_90_text}")
print(f"- 분산의 95%를 설명하는 주성분 개수: {variance_95_text}")

plt.figure(figsize=(9, 5.5))
plt.plot(range(1, max_variance_components + 1), cumulative_variance, color='darkviolet', lw=2.2, label='Cumulative Explained Variance')
label_90 = f'{n_components_90} PCs' if n_components_90 is not None else f'not reached by {max_variance_components} PCs'
label_95 = f'{n_components_95} PCs' if n_components_95 is not None else f'not reached by {max_variance_components} PCs'
plt.axhline(y=0.90, color='red', linestyle='--', label=f'90% Variance threshold ({label_90})')
if n_components_90 is not None:
    plt.axvline(x=n_components_90, color='red', linestyle=':', alpha=0.7)
plt.axhline(y=0.95, color='forestgreen', linestyle='--', label=f'95% Variance threshold ({label_95})')
if n_components_95 is not None:
    plt.axvline(x=n_components_95, color='forestgreen', linestyle=':', alpha=0.7)
plt.xlabel("Number of Principal Components", fontsize=11)
plt.ylabel("Cumulative Explained Variance Ratio", fontsize=11)
plt.title("Cumulative Explained Variance Plot", fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "fig5_cumulative_explained_variance.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"  -> Figure 5 저장 완료: {fig5_path}")


# ====================================================
# 4. PCA + KNN 분류 실험
# ====================================================
print("\n" + "=" * 60)
print("[Step 4] PCA + KNN 분류기 평가 (주성분 수 및 화이트닝 비교)")
print("=" * 60)

component_list = [10, 30, 50, 75, 100, 150, 200, 250, 300]
knn_values = [1, 3, 5]
whitening_values = [False, True]
knn_accuracies = {
    (whiten, k_val): []
    for whiten in whitening_values
    for k_val in knn_values
}

for n_comp in component_list:
    for whiten in whitening_values:
        pca = PCA(
            n_components=n_comp,
            whiten=whiten,
            random_state=RANDOM_SEED
        ).fit(X_train)
        X_train_projected = pca.transform(X_train)
        X_test_projected = pca.transform(X_test)

        for k_val in knn_values:
            knn = KNeighborsClassifier(n_neighbors=k_val)
            knn.fit(X_train_projected, y_train)
            accuracy = accuracy_score(y_test, knn.predict(X_test_projected))
            knn_accuracies[(whiten, k_val)].append(accuracy)

# [Figure 6] 모든 k와 whitening 조합에 대한 주성분 수별 KNN 테스트 정확도
plt.figure(figsize=(11, 7))
colors = {False: 'royalblue', True: 'darkorange'}
markers = {1: 'o', 3: 's', 5: '^'}
for whiten in whitening_values:
    for k_val in knn_values:
        label = f"{'With' if whiten else 'Without'} Whitening (k={k_val})"
        plt.plot(
            component_list,
            knn_accuracies[(whiten, k_val)],
            marker=markers[k_val],
            lw=1.8,
            color=colors[whiten],
            linestyle='-' if k_val == 1 else ('--' if k_val == 3 else ':'),
            label=label
        )
plt.xlabel("Number of Principal Components", fontsize=11)
plt.ylabel("Test Accuracy", fontsize=11)
plt.title("KNN Test Accuracy vs. Number of Principal Components", fontsize=13)
plt.legend(loc='best', fontsize=9, ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig6_path = os.path.join(OUTPUT_DIR, "fig6_knn_accuracy_vs_components.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"  -> Figure 6 저장 완료: {fig6_path}")

# 전체 실험 중 테스트 정확도가 가장 높은 PCA + KNN 설정을 기록
best_pca_knn_config = max(
    (
        (accuracy, n_comp, whiten, k_val)
        for whiten in whitening_values
        for k_val in knn_values
        for n_comp, accuracy in zip(component_list, knn_accuracies[(whiten, k_val)])
    ),
    key=lambda result: result[0]
)
best_pca_knn_acc, best_pca_knn_components, best_pca_knn_whiten, best_pca_knn_k = best_pca_knn_config
print("\n- KNN 정확도 요약 (100 Components):")
for whiten in whitening_values:
    for k_val in knn_values:
        accuracy = knn_accuracies[(whiten, k_val)][component_list.index(100)]
        print(f"  whitening={whiten}, k={k_val}: Test Accuracy = {accuracy * 100:.2f}%")
print(
    f"\n- PCA + KNN 최고 테스트 정확도: {best_pca_knn_acc * 100:.2f}% "
    f"(components={best_pca_knn_components}, whitening={best_pca_knn_whiten}, k={best_pca_knn_k})"
)


# ====================================================
# 5. CNN 모델 훈련 및 평가
# ====================================================
print("\n" + "=" * 60)
print("[Step 5] Convolutional Neural Network (CNN) 훈련")
print("=" * 60)

# CNN 검증 데이터는 학습 세트에서 분리하고, 원래 테스트 세트는 최종 평가에만 사용
cnn_img_train, cnn_img_val, cnn_y_train, cnn_y_val = train_test_split(
    img_train,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=RANDOM_SEED
)

# CNN 4차원 텐서 변환: (N, 87, 65, 1)
cnn_X_train = cnn_img_train.reshape(-1, h, w, 1)
cnn_X_val = cnn_img_val.reshape(-1, h, w, 1)
cnn_X_test = img_test.reshape(-1, h, w, 1)

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn_model = create_cnn_model((h, w, 1), n_classes)
cnn_model.summary()

# CNN 모델 학습
epochs = 40
batch_size = 32
history = cnn_model.fit(
    cnn_X_train, cnn_y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(cnn_X_val, cnn_y_val),
    verbose=1
)

# [Figure 7] CNN 에포크별 Train/Validation Accuracy 및 Loss 곡선
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history.history['accuracy'], color='royalblue', lw=2, label='Training Accuracy')
plt.plot(range(1, epochs + 1), history.history['val_accuracy'], color='crimson', lw=2, label='Validation Accuracy')
plt.title("CNN Training and Validation Accuracy", fontsize=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history.history['loss'], color='royalblue', lw=2, label='Training Loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], color='crimson', lw=2, label='Validation Loss')
plt.title("CNN Training and Validation Loss", fontsize=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig7_path = os.path.join(OUTPUT_DIR, "fig7_cnn_curves.png")
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"  -> Figure 7 저장 완료: {fig7_path}")

cnn_loss, cnn_test_acc = cnn_model.evaluate(cnn_X_test, y_test, verbose=0)
print(f"\n- CNN 최종 Test Loss: {cnn_loss:.4f}")
print(f"- CNN 최종 Test Accuracy: {cnn_test_acc * 100:.2f}%")


# ====================================================
# 6. 최종 성능 비교 (PCA + KNN vs CNN)
# ====================================================
print("\n" + "=" * 60)
print("[Step 6] 최종 성능 비교 (PCA + KNN vs CNN)")
print("=" * 60)

# [Figure 8] 최종 정확도 비교: PCA + KNN vs CNN
best_knn_label = (
    f'PCA + KNN (best: {best_pca_knn_components} PCs, '
    f"{'whitened' if best_pca_knn_whiten else 'no whitening'}, k={best_pca_knn_k})"
)
models = [best_knn_label, 'CNN']
accuracies = [best_pca_knn_acc * 100, cnn_test_acc * 100]

plt.figure(figsize=(7, 5.5))
bars = plt.bar(models, accuracies, color=['#4C72B0', '#DD8452'], width=0.45)
plt.ylabel("Test Accuracy (%)", fontsize=11)
plt.ylim(0, 100)
plt.title("Final Accuracy Comparison: PCA + KNN vs. CNN", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1.5, f"{height:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
fig8_path = os.path.join(OUTPUT_DIR, "fig8_final_accuracy_comparison.png")
plt.savefig(fig8_path, dpi=300)
plt.close()
print(f"  -> Figure 8 저장 완료: {fig8_path}")

print("\n" + "=" * 60)
print("모든 실험 및 8개 필수 결과 그림 저장이 완료되었습니다!")
print(f"결과 이미지 저장 위치: {OUTPUT_DIR}")
print("=" * 60)
