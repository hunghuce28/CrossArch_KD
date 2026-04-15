"""
==========================================================================
KIỂM TRA OVERFITTING CỦA ViT-B/16 TEACHER (Chạy trên Colab)
==========================================================================
Script này đánh giá chất lượng Teacher ViT-B/16 trên cả train và val set:

1. Confusion Matrix        → Model có thiên lệch class nào không?
2. Per-class Precision/Recall/F1 → Class nào yếu nhất?
3. Train Acc vs Val Acc    → Khoảng cách = mức overfit
4. Confidence Distribution → Model có tự tin "đúng" quá mức không?
5. Visualization sai lầm   → Xem các ảnh bị phân loại sai

Kết quả ban đầu (README):
  - Train Acc: 99.1%
  - Val Acc:   99.14%
  - Gap:       ~0% → CÓ VẺ KHÔNG OVERFIT
  
Nhưng cần kiểm tra sâu hơn vì:
  - Dataset nhỏ 4 classes + ViT-B/16 (86M params) → model quá lớn cho task
  - Val set có thể quá giống train set (cùng nguồn, cùng điều kiện chụp)
  - 99%+ accuracy trên 4 classes → nghi ngờ data leakage
==========================================================================
"""

# ====== DÁN TOÀN BỘ CODE NÀY VÀO 1 CELL COLAB ======

# %% [markdown]
# ## Kiểm Tra Overfitting ViT-B/16

# %%
# ====== CELL: KIỂM TRA OVERFITTING VIT-B/16 ======
# %cd /content/CrossArch_KD/CAKD
# Cài đặt
# !pip install scikit-learn matplotlib seaborn

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Colab backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CẤU HÌNH
# ================================================================
DATA_PATH = "/content/TRASH_Dataset_Split"
TEACHER_CKPT = "/content/CrossArch_KD/CAKD/results/vit_b16_teacher/best_teacher.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_WORKERS = 2
SAVE_DIR = "overfit_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 65)
print("  🔍 KIỂM TRA OVERFITTING: ViT-B/16 Teacher")
print("=" * 65)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n📂 Loading data...")

# Transform cho evaluation (không augment)
eval_transform = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(
    os.path.join(DATA_PATH, "train"), transform=eval_transform
)
val_dataset = torchvision.datasets.ImageFolder(
    os.path.join(DATA_PATH, "val"), transform=eval_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS
)

# Phân bố class
train_counts = Counter([label for _, label in train_dataset.samples])
val_counts = Counter([label for _, label in val_dataset.samples])

print(f"   Classes: {class_names}")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples:   {len(val_dataset)}")
print(f"\n   Phân bố Train:")
for idx, name in enumerate(class_names):
    print(f"     {name}: {train_counts[idx]} ({train_counts[idx]/len(train_dataset)*100:.1f}%)")
print(f"\n   Phân bố Val:")
for idx, name in enumerate(class_names):
    print(f"     {name}: {val_counts[idx]} ({val_counts[idx]/len(val_dataset)*100:.1f}%)")

# ================================================================
# 2. LOAD MODEL
# ================================================================
print(f"\n🧠 Loading teacher model from: {TEACHER_CKPT}")

model = torchvision.models.vit_b_16()
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
ckpt = torch.load(TEACHER_CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
model.eval()

epoch_trained = ckpt.get('epoch', 'N/A')
print(f"   Loaded from epoch: {epoch_trained}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Params / sample: {sum(p.numel() for p in model.parameters()) / len(train_dataset):.0f}")
print(f"   → Nếu params/sample > 1000: model quá lớn cho dataset = dễ overfit")

# ================================================================
# 3. HÀM INFERENCE
# ================================================================
@torch.no_grad()
def full_inference(model, dataloader, device):
    """Chạy inference và trả về predictions, labels, probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        if isinstance(output, tuple):
            output = output[0]

        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
        losses = criterion(output, labels)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        all_losses.append(losses.cpu())

    return {
        'preds': torch.cat(all_preds).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'probs': torch.cat(all_probs).numpy(),
        'losses': torch.cat(all_losses).numpy(),
    }

# ================================================================
# 4. CHẠY INFERENCE
# ================================================================
print("\n⏳ Running inference on train set...")
train_results = full_inference(model, train_loader, DEVICE)
train_acc = (train_results['preds'] == train_results['labels']).mean() * 100

print("⏳ Running inference on val set...")
val_results = full_inference(model, val_loader, DEVICE)
val_acc = (val_results['preds'] == val_results['labels']).mean() * 100

# ================================================================
# 5. KẾT QUẢ TỔNG QUAN
# ================================================================
gap = train_acc - val_acc
train_loss = train_results['losses'].mean()
val_loss = val_results['losses'].mean()
loss_ratio = val_loss / train_loss if train_loss > 0 else float('inf')

print("\n" + "=" * 65)
print("  📊 PHÂN TÍCH OVERFITTING")
print("=" * 65)

print(f"\n  {'Metric':<25} {'Train':>12} {'Val':>12} {'Gap':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Accuracy':<25} {train_acc:>11.2f}% {val_acc:>11.2f}% {gap:>+11.2f}%")
print(f"  {'Loss (CE)':<25} {train_loss:>12.4f} {val_loss:>12.4f} {val_loss-train_loss:>+12.4f}")
print(f"  {'Loss ratio (val/train)':<25} {'':>12} {'':>12} {loss_ratio:>12.2f}x")

# Đánh giá overfit
print(f"\n  🔎 ĐÁNH GIÁ:")
if gap < 2:
    print(f"  ✅ Gap accuracy = {gap:.2f}% → KHÔNG overfit rõ ràng")
elif gap < 5:
    print(f"  ⚠️  Gap accuracy = {gap:.2f}% → Overfit NHẸ")
elif gap < 10:
    print(f"  🟡 Gap accuracy = {gap:.2f}% → Overfit TRUNG BÌNH")
else:
    print(f"  🔴 Gap accuracy = {gap:.2f}% → Overfit NẶNG")

if loss_ratio > 3:
    print(f"  🔴 Loss ratio = {loss_ratio:.1f}x → Model tự tin SAI trên val (overfit)")
elif loss_ratio > 1.5:
    print(f"  ⚠️  Loss ratio = {loss_ratio:.1f}x → Có dấu hiệu overfit qua loss")
else:
    print(f"  ✅ Loss ratio = {loss_ratio:.1f}x → Loss ổn định")

params_per_sample = sum(p.numel() for p in model.parameters()) / len(train_dataset)
if params_per_sample > 5000:
    print(f"  🔴 Params/sample = {params_per_sample:.0f} → Model QUÁ LỚN cho dataset!")
    print(f"     ViT-B/16 (86M params) trên {len(train_dataset)} ảnh → dễ memorize")
elif params_per_sample > 1000:
    print(f"  ⚠️  Params/sample = {params_per_sample:.0f} → Model lớn, cần theo dõi")

# ================================================================
# 6. PER-CLASS METRICS
# ================================================================
print(f"\n  📋 CHI TIẾT TỪNG CLASS (Validation Set):")
print(f"  {'-'*65}")

report = classification_report(
    val_results['labels'], val_results['preds'],
    target_names=class_names, output_dict=True
)

print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for cls_name in class_names:
    r = report[cls_name]
    print(f"  {cls_name:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
          f"{r['f1-score']:>10.4f} {r['support']:>10.0f}")

# Kiểm tra class bị nhầm lẫn
print(f"\n  📋 CHI TIẾT TỪNG CLASS (Train Set):")
print(f"  {'-'*65}")

train_report = classification_report(
    train_results['labels'], train_results['preds'],
    target_names=class_names, output_dict=True
)

print(f"  {'Class':<20} {'Train Acc':>10} {'Val Acc':>10} {'Gap':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
for idx, cls_name in enumerate(class_names):
    t_recall = train_report[cls_name]['recall'] * 100
    v_recall = report[cls_name]['recall'] * 100
    g = t_recall - v_recall
    flag = " ⚠️" if g > 5 else ""
    print(f"  {cls_name:<20} {t_recall:>9.1f}% {v_recall:>9.1f}% {g:>+9.1f}%{flag}")

# ================================================================
# 7. CONFIDENCE ANALYSIS
# ================================================================
print(f"\n  📊 PHÂN TÍCH CONFIDENCE:")

# Confidence của dự đoán đúng vs sai
val_probs = val_results['probs']
val_preds = val_results['preds']
val_labels = val_results['labels']
val_max_probs = val_probs.max(axis=1)

correct_mask = val_preds == val_labels
wrong_mask = ~correct_mask

print(f"  {'':>30} {'Mean':>8} {'Min':>8} {'Max':>8}")
print(f"  {'Confidence (đúng)':<30} {val_max_probs[correct_mask].mean():>8.4f} "
      f"{val_max_probs[correct_mask].min():>8.4f} {val_max_probs[correct_mask].max():>8.4f}")

if wrong_mask.sum() > 0:
    print(f"  {'Confidence (sai)':<30} {val_max_probs[wrong_mask].mean():>8.4f} "
          f"{val_max_probs[wrong_mask].min():>8.4f} {val_max_probs[wrong_mask].max():>8.4f}")
    if val_max_probs[wrong_mask].mean() > 0.8:
        print(f"  🔴 Model TỰ TIN SAI: confidence dự đoán sai = {val_max_probs[wrong_mask].mean():.1%}")
        print(f"     → Dấu hiệu overfit: model memorize pattern sai")
else:
    print(f"  {'Confidence (sai)':<30} {'N/A':>8} (không có mẫu nào sai)")

# Mean confidence trên train
train_max_probs = train_results['probs'].max(axis=1)
print(f"\n  {'Mean confidence (train)':<30} {train_max_probs.mean():>8.4f}")
print(f"  {'Mean confidence (val)':<30} {val_max_probs.mean():>8.4f}")

if train_max_probs.mean() > 0.99:
    print(f"  ⚠️  Train confidence > 99% → Model có thể đã memorize training data")

# ================================================================
# 8. VẼ BIỂU ĐỒ
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Overfitting Analysis: ViT-B/16 Teacher', fontsize=16, fontweight='bold')

# 8a. Confusion Matrix (Val)
cm_val = confusion_matrix(val_labels, val_preds)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix (Validation)', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')

# 8b. Confusion Matrix (Train)
cm_train = confusion_matrix(train_results['labels'], train_results['preds'])
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix (Train)', fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

# 8c. Per-class Accuracy: Train vs Val
x = np.arange(num_classes)
width = 0.35

train_accs = [train_report[c]['recall'] * 100 for c in class_names]
val_accs = [report[c]['recall'] * 100 for c in class_names]

bars1 = axes[1, 0].bar(x - width/2, train_accs, width, label='Train', color='#ff7043')
bars2 = axes[1, 0].bar(x + width/2, val_accs, width, label='Val', color='#42a5f5')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Per-class Accuracy: Train vs Val', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].set_ylim([max(0, min(min(train_accs), min(val_accs)) - 5), 102])

# Annotate bars
for bar in bars1:
    axes[1, 0].annotate(f'{bar.get_height():.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
for bar in bars2:
    axes[1, 0].annotate(f'{bar.get_height():.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

# 8d. Confidence Distribution
axes[1, 1].hist(train_max_probs, bins=50, alpha=0.6, label='Train',
                color='#ff7043', density=True)
axes[1, 1].hist(val_max_probs, bins=50, alpha=0.6, label='Val',
                color='#42a5f5', density=True)
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random guess')
axes[1, 1].set_xlabel('Max Prediction Confidence')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Confidence Distribution', fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, 'overfit_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n📊 Biểu đồ đã lưu tại: {save_path}")

# ================================================================
# 9. PHÂN TÍCH CÁC MẪU BỊ DỰ ĐOÁN SAI
# ================================================================
wrong_indices = np.where(wrong_mask)[0]

if len(wrong_indices) > 0:
    print(f"\n  ❌ CÁC MẪU BỊ DỰ ĐOÁN SAI TRÊN VAL ({len(wrong_indices)} mẫu):")
    print(f"  {'-'*65}")
    print(f"  {'#':>4} {'True Label':<20} {'Predicted':<20} {'Confidence':>10}")
    print(f"  {'-'*4} {'-'*20} {'-'*20} {'-'*10}")
    
    for i, idx in enumerate(wrong_indices[:20]):  # Hiện tối đa 20 mẫu
        true_cls = class_names[val_labels[idx]]
        pred_cls = class_names[val_preds[idx]]
        conf = val_max_probs[idx]
        print(f"  {i+1:>4} {true_cls:<20} {pred_cls:<20} {conf:>10.4f}")
    
    if len(wrong_indices) > 20:
        print(f"  ... và {len(wrong_indices) - 20} mẫu khác")
    
    # Ma trận nhầm lẫn chi tiết cho các mẫu sai
    print(f"\n  📋 NHẦM LẪN GIỮA CÁC CLASS:")
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_val[i][j] > 0:
                print(f"     {class_names[i]:>15} → {class_names[j]:<15}: "
                      f"{cm_val[i][j]} mẫu ({cm_val[i][j]/cm_val[i].sum()*100:.1f}%)")
else:
    print(f"\n  ✅ Không có mẫu nào bị dự đoán sai trên val set!")
    print(f"  ⚠️  CẢNH BÁO: 100% accuracy trên val → có thể:")
    print(f"     1. Val set quá nhỏ hoặc quá dễ")
    print(f"     2. Val set bị trùng/tương tự train set (data leakage)")
    print(f"     3. Task 4 classes quá đơn giản cho ViT-B/16 (86M params)")

# ================================================================
# 10. ĐỀ XUẤT
# ================================================================
print(f"\n" + "=" * 65)
print(f"  💡 ĐỀ XUẤT")
print(f"=" * 65)

if val_acc > 98 and gap < 2:
    print(f"""
  Mặc dù gap thấp (Train {train_acc:.1f}% vs Val {val_acc:.1f}%), nhưng:

  1. ⚠️  ViT-B/16 (86M params) trên {len(train_dataset)} ảnh 4 classes
     → Tỷ lệ params/sample = {params_per_sample:.0f} (quá cao!)
     → Model có thể memorize cả train lẫn val vì chúng cùng distribution

  2. ⚠️  Để kiểm tra thật:
     a) Test trên ảnh HOÀN TOÀN MỚI (chụp khác, nguồn khác)
     b) Dùng K-Fold Cross Validation
     c) Kiểm tra val set có bị overlap với train không

  3. ✅ Cho mục đích CAKD (KD), Teacher 99% là TỐT:
     → Teacher càng mạnh, Student (ResNet50) càng học được nhiều
     → Kể cả Teacher overfit nhẹ, KD vẫn có lợi vì:
        - Soft labels từ Teacher chứa "dark knowledge"
        - Student có capacity nhỏ hơn nên không bắt chước được memorization
""")

# Copy biểu đồ lên Drive
print("  📊 Để lưu biểu đồ vào Drive:")
print(f"     !cp {save_path} /content/drive/MyDrive/")
print()
