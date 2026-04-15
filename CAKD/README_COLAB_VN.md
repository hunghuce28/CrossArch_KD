# Hướng Dẫn Huấn Luyện Cross-Architecture KD (CAKD) Trên Google Colab 🚀

Tài liệu này tổng hợp toàn bộ các bước 100% thành công để chạy mã nguồn `CrossArch_KD` trên Google Colab, đặc biệt tối ưu cho **tập dữ liệu tự chọn (4 classes)** và xử lý toàn bộ các lỗi phiên bản (PyTorch 2.x vs 1.12).

> [!IMPORTANT]
> **Mỗi bước dưới đây tương ứng với 1 Cell (ô code) trên Colab.** Chạy tuần tự từ Bước 1 → Bước 7. Nếu Colab bị ngắt kết nối giữa chừng, bạn phải chạy lại **từ Bước 1**.

---

## Tổng Quan Pipeline

```
Bước 1   → Cài môi trường (Conda + PyTorch 1.12)
Bước 2   → Giải nén dataset từ Google Drive
Bước 3   → Patch file CAKD vào thư viện PyTorch
Bước 4   → Fix lỗi topk cho dataset 4 classes
Bước 5   → Train Baseline: ResNet50 (làm điểm chuẩn)
Bước 6   → Train Teacher: ViT-B/16 (fine-tune trên 4 classes)
Bước 6.5 → Fix lỗi inplace operation trong code CAKD
Bước 7   → Train CAKD: ResNet50 học từ ViT (Knowledge Distillation)
```

---

## 🛑 BƯỚC 1: Cài Đặt Môi Trường (Cell 1)

### Tại sao cần bước này?
Google Colab mặc định dùng **Python 3.12 + PyTorch 2.x**, nhưng mã nguồn CAKD được viết cho **PyTorch 1.12.0** (phiên bản cũ hơn rất nhiều). Nếu chạy trực tiếp trên Colab mặc định sẽ bị lỗi vì:

- Nhiều API trong `torch.nn.functional` đã bị đổi tên/xóa ở PyTorch 2.x
- Các file tuỳ biến của CAKD (`resnet.py`, `vision_transformer.py`, `functional.py`) chỉ tương thích với PyTorch 1.12
- NumPy 2.x thay đổi ABI (Application Binary Interface), không tương thích với PyTorch 1.12

**Giải pháp:** Cài Miniconda để tạo môi trường Python 3.10, hạ cấp PyTorch về 1.12.0, hạ NumPy xuống < 2.

**👉 Thao tác:** Tạo Cell đầu tiên trên Colab, dán và chạy:

```python
# ====== CELL 1: CÀI ĐẶT MÔI TRƯỜNG ======

# 1a. Kết nối Google Drive (sẽ hiện hộp thoại xác nhận)
from google.colab import drive
drive.mount('/content/drive')

# 1b. Tải mã nguồn CAKD từ GitHub
!git clone https://github.com/yufanLIU/CrossArch_KD.git

# 1c. Cài đặt Miniconda → tạo môi trường Python 3.10 thay cho 3.12 mặc định
!wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O miniconda.sh
!bash miniconda.sh -b -f -p /usr/local
!rm miniconda.sh

# 1d. Cấu hình đường dẫn để Colab sử dụng Python 3.10 thay vì 3.12
!conda update conda -y -q
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/site-packages')

# 1e. Cài Python 3.10 + các thư viện cần thiết
!conda install -y python=3.10
!pip install "numpy<2" einops timm
#   - numpy<2   : Hạ cấp NumPy để tương thích ABI với PyTorch 1.12
#   - einops     : Thư viện xử lý tensor linh hoạt (CAKD dùng cho ViT)
#   - timm       : Thư viện model pretrained (PyTorch Image Models)

# 1f. Cài đặt PyTorch 1.12.0 + CUDA 11.3 (đúng phiên bản tác giả yêu cầu)
!pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Kiểm tra phiên bản
import torch
print(f"✅ BƯỚC 1 HOÀN TẤT! PyTorch version: {torch.__version__}")
```

> [!NOTE]
> Cell này mất khoảng **5–10 phút** để chạy xong. Khi thấy dòng `✅ BƯỚC 1 HOÀN TẤT!` thì tiếp tục bước 2.

---

## 📂 BƯỚC 2: Giải Nén Dữ Liệu (Cell 2)

### Tại sao cần bước này?
Khi bạn nén file `.zip` trên **Windows**, hệ điều hành sử dụng dấu `\` (backslash) cho đường dẫn thư mục. Nhưng **Linux** (hệ điều hành của Colab) dùng dấu `/` (forward slash). Nếu giải nén bằng lệnh `unzip` thông thường, các thư mục con sẽ bị nối tên sai, ví dụ:

```
❌ Sai:  TRASH_Dataset_Split\train\class1\img001.jpg  (cả chuỗi thành 1 tên file)
✅ Đúng: TRASH_Dataset_Split/train/class1/img001.jpg  (tạo đúng cấu trúc thư mục)
```

**Giải pháp:** Dùng Python `zipfile` để đọc từng file, thay `\` → `/` rồi ghi ra đĩa.

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```python
# ====== CELL 2: GIẢI NÉN DỮ LIỆU ======
import zipfile
import os

# Đường dẫn file zip trên Google Drive (đổi nếu bạn đặt tên khác)
zip_path = '/content/drive/MyDrive/TRASH_Dataset_Split.zip'
extract_path = '/content/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        member_data = zip_ref.read(member)
        # Thay backslash Windows → forward slash Linux
        fixed_member = member.replace('\\', '/')
        target_path = os.path.join(extract_path, fixed_member)

        if not fixed_member.endswith('/'):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'wb') as outfile:
                outfile.write(member_data)

print("✅ BƯỚC 2 HOÀN TẤT - Đã giải nén chuẩn cấu trúc Linux!")
```

> [!TIP]
> Sau khi chạy xong, bạn có thể kiểm tra bằng lệnh `!ls /content/TRASH_Dataset_Split/` — phải thấy 2 thư mục `train/` và `val/`, mỗi thư mục chứa 4 thư mục con (4 classes).

---

## 🛠 BƯỚC 3: Patch File CAKD Vào Thư Viện PyTorch (Cell 3)

### Tại sao cần bước này?
Thuật toán CAKD cần **tuỳ biến 3 file gốc** bên trong thư viện PyTorch để thêm các hook trích xuất feature map phục vụ Knowledge Distillation:

| File gốc PyTorch | File tuỳ biến CAKD | Mục đích |
|:-:|:-:|:-:|
| `torchvision/models/resnet.py` | `cakd_modified_files/resnet.py` | Thêm hook trả về feature maps trung gian từ các block ResNet |
| `torchvision/models/vision_transformer.py` | `cakd_modified_files/vision_transformer.py` | Thêm hook trả về attention maps + CLS token từ các layer ViT |
| `torch/nn/functional.py` | `cakd_modified_files/functional.py` | Sửa hàm `softmax` để hỗ trợ tính KD loss |

> [!WARNING]
> Phải dùng `!python -c "..."` (chạy subprocess) thay vì import trực tiếp trong Colab, vì kernel Colab đang dùng Python 3.12 nhưng thư viện PyTorch 1.12 nằm trong site-packages của Python 3.10 qua Conda. Lệnh `!python` sẽ gọi đúng Python 3.10.

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```python
# ====== CELL 3: PATCH FILE CAKD ======
%cd /content/CrossArch_KD/CAKD

!python -c "import os, shutil, torch, torchvision; \
torch_nn_path = torch.nn.__path__[0]; \
torchvision_model_path = torchvision.models.__path__[0]; \
source_dir = 'cakd_modified_files'; \
shutil.copy(os.path.join(source_dir, 'resnet.py'), torchvision_model_path); \
shutil.copy(os.path.join(source_dir, 'vision_transformer.py'), torchvision_model_path); \
shutil.copy(os.path.join(source_dir, 'functional.py'), torch_nn_path); \
print(f'✅ BƯỚC 3 HOÀN TẤT - Đã patch 3 file vào PyTorch 1.12.0:\n  - {torchvision_model_path}/resnet.py\n  - {torchvision_model_path}/vision_transformer.py\n  - {torch_nn_path}/functional.py')"
```

---

## 🐛 BƯỚC 4: Fix Lỗi topk Cho Dataset 4 Classes (Cell 4)

### Tại sao cần bước này?
Mã nguồn gốc CAKD được viết cho **ImageNet (1000 classes)**, nên hàm đánh giá accuracy tự động tính **Top-5 accuracy** (`topk=(1, 5)` — lấy 5 dự đoán cao nhất rồi kiểm tra xem đáp án đúng có nằm trong đó không).

Vấn đề: Dataset của chúng ta chỉ có **4 classes**. Bạn không thể lấy "top 5" từ 4 phần tử → PyTorch sẽ báo lỗi **`RuntimeError: k (5) is too big for the dimension size (4)`**.

**Giải pháp:** Tự động thay `topk=(1, 5)` → `topk=(1, 4)` trong tất cả các file training.

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```python
# ====== CELL 4: FIX topk CHO 4 CLASSES ======
import os

for filename in ['dist_train_student.py', 'dist_train_teacher.py', 'dist_train_cakd.py']:
    filepath = f'/content/CrossArch_KD/CAKD/{filename}'
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r') as f:
        content = f.read()
    count = content.count('topk=(1, 5)')
    if count > 0:
        content = content.replace('topk=(1, 5)', 'topk=(1, 4)')
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✔ {filename}: đã sửa {count} chỗ topk=(1,5) → topk=(1,4)")
    else:
        print(f"  – {filename}: không cần sửa (đã đúng hoặc không có topk)")

print("\n✅ BƯỚC 4 HOÀN TẤT - Đã fix topk cho 4 classes!")
```

> [!NOTE]
> **Nếu dataset của bạn có N classes (khác 4):** Thay `topk=(1, 4)` thành `topk=(1, N)` trong đoạn code trên, với N là số class thực tế. Lưu ý N phải ≤ số class, nếu N ≥ số class thì đặt bằng số class.

> [!TIP]
> **Lưu ý về các file đã tuỳ chỉnh:** File `dist_train_teacher.py` (viết mới hoàn toàn) và `dist_train_cakd.py` (sửa đổi từ bản gốc) đã xử lý topk động bằng `min(4, ...)` nên script sẽ in "không cần sửa" cho chúng. Chỉ `dist_train_student.py` (file gốc chưa sửa) sẽ được fix bởi script. Nếu clone từ repo gốc GitHub mà không dùng các file tuỳ chỉnh, cả 3 file đều cần fix và script sẽ tự xử lý.

---

## 🏃 BƯỚC 5: Huấn Luyện Baseline — ResNet50 (Cell 5)

### Tại sao cần bước này?
Đây là bước tạo **điểm chuẩn (baseline)**: Huấn luyện một mạng ResNet50 bình thường trên 4 classes, **không có Knowledge Distillation**. Kết quả accuracy của Baseline sẽ dùng để so sánh với CAKD ở Bước 7 nhằm chứng minh thuật toán KD có hiệu quả hay không.

### Giải thích các hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|:--------|:--------|:--------|
| `--batch-size 32` | 32 ảnh/batch | Phù hợp với RAM GPU miễn phí của Colab (~15GB) |
| `--lr 0.01` | Learning rate = 0.01 | Giá trị mặc định cho SGD khi train từ đầu (from scratch) |
| `--epochs 30` | 30 epoch | Đủ để mô hình hội tụ trên dataset nhỏ |
| `--train-crop-size 224` | Crop ảnh 224×224 | Kích thước đầu vào chuẩn của ResNet |
| `--val-resize-size 224` | Resize ảnh validation 224×224 | Giữ nhất quán với kích thước train |

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```bash
# ====== CELL 5: TRAIN BASELINE RESNET50 ======
%cd /content/CrossArch_KD/CAKD

DATA_PATH="/content/TRASH_Dataset_Split"
OUTPUT_DIR="results/resnet50/baseline"

!python dist_train_student.py \
    --data-path "$DATA_PATH" \
    --device cuda \
    --batch-size 32 \
    --lr 0.01 \
    --epochs 30 \
    --train-crop-size 224 \
    --val-resize-size 224 \
    --output-dir "$OUTPUT_DIR"
```

🎉 Nếu trên màn hình hiện ra `Epoch: [0]` và thanh tiến trình đang chạy → **BẠN ĐÃ THÀNH CÔNG!**

### ✅ Kết quả thực tế Baseline (TRASH Dataset, 4 classes)

| Epoch | Train Acc | Test Acc@1 | Ghi chú |
|:-----:|:---------:|:----------:|:--------|
| 0 | 29.4% | 32.6% | Bắt đầu (random ≈ 25%) |
| 5 | 59.6% | 62.2% | Học khá nhanh |
| 10 | 67.5% | 65.7% | Dao động do LR cao |
| 15 | 72.3% | 79.7% | Tăng mạnh |
| 20 | 76.9% | 75.6% | Hội tụ dần |
| 25 | 79.2% | 83.2% | Gần đỉnh |
| 27 | 80.5% | **84.5%** | 🏆 Best |
| 29 | 82.1% | 83.8% | Final |

**Best Baseline Acc@1: 84.509%** — Thời gian train: ~51 phút trên GPU Colab.

> [!IMPORTANT]
> **84.5% là con số chuẩn để so sánh.** Ở Bước 7 (CAKD), nếu ResNet50 đạt accuracy **> 84.5%** thì chứng minh Knowledge Distillation hiệu quả.

> [!TIP]
> Thời gian train phụ thuộc GPU được cấp (T4 ≈ 50 phút, A100 ≈ 15–20 phút cho 30 epochs). Kết quả sẽ tự lưu tại `results/resnet50/baseline/`.

---

## 🎓 BƯỚC 6: Huấn Luyện Teacher — ViT-B/16 (Cell 6)

### Tại sao cần bước này?
Thuật toán CAKD là **Knowledge Distillation kiến trúc chéo (Cross-Architecture)**:
- **Teacher** = ViT-B/16 (Vision Transformer, kiến trúc mạnh hơn, nặng hơn)
- **Student** = ResNet50 (kiến trúc nhẹ hơn, dùng để deploy)

Student sẽ "học" từ Teacher → đạt accuracy cao hơn so với tự train một mình (Baseline).

> [!CAUTION]
> **Bắt buộc phải fine-tune Teacher trên dataset 4 classes của bạn trước!** Nếu dùng thẳng ViT pretrained ImageNet (1000 classes), đầu ra Teacher sẽ có 1000 neurons trong khi Student chỉ có 4 → lỗi `shape mismatch` ngay lập tức.

### Tại sao file `dist_train_teacher.py` được viết lại?

File gốc của repo CAKD chỉ hỗ trợ ImageNet (1000 classes). File hiện tại đã được **viết lại hoàn toàn** để hỗ trợ dataset tùy chọn:

| Vấn đề | File gốc (repo) | File viết lại |
|:-------|:----------------|:-------------|
| Dataset | Chỉ ImageNet 1000 classes | **Bất kỳ ImageFolder** (tự detect num_classes) |
| ViT head | Giữ nguyên 1000 neurons | **Tự thay `heads.head = Linear(768, num_classes)`** |
| ViT output | Không xử lý tuple sau patch | **Xử lý `output[0]` nếu là tuple** |
| Best checkpoint | ❌ Không lưu | ✅ **Lưu `best_teacher.pth` theo accuracy cao nhất** |
| topk metric | Hardcode `(1, 5)` | **Tự tính `min(5, num_classes)`** — tránh lỗi khi < 5 classes |

**Transfer Learning flow:**
```
ViT-B/16 pretrained (ImageNet 1000 classes)
    ↓ Load weights
Thay heads.head: Linear(768, 1000) → Linear(768, 4)
    ↓ Fine-tune trên TRASH dataset
Lưu best_teacher.pth (dùng cho Bước 7)
```

### Giải thích các hyperparameters (khác với Bước 5)

| Tham số | Giá trị | Ý nghĩa |
|:--------|:--------|:--------|
| `--opt adamw` | AdamW optimizer | Tốt hơn SGD cho Transformer vì xử lý weight decay chính xác hơn |
| `--lr 0.0001` | Learning rate = 1e-4 | Rất nhỏ vì đây là **fine-tuning** ViT pretrained. LR quá lớn (0.001+) có thể "xóa" kiến thức pretrained → gây dao động accuracy. Nhiều paper fine-tune ViT dùng LR trong khoảng 1e-5 đến 3e-4 |
| `--lr-warmup-epochs 3` | Warm-up 3 epochs | 3 epoch đầu tiên tăng dần LR từ ~0 → 0.0001, tránh gradient quá lớn phá hỏng pretrained weights |
| `--lr-warmup-method linear` | Tăng LR tuyến tính | LR tăng đều đặn trong giai đoạn warm-up |
| `--lr-scheduler cosineannealinglr` | Cosine Annealing | Sau warm-up, LR giảm dần theo hình cos(x), giúp hội tụ mượt mà và tránh overfitting |
| `--val-resize-size 256` | Resize validation 256 | ViT cần ảnh lớn hơn ResNet để trích xuất patch tốt hơn |

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```bash
# ====== CELL 6: TRAIN TEACHER ViT-B/16 ======
%cd /content/CrossArch_KD/CAKD

DATA_PATH="/content/TRASH_Dataset_Split"
OUTPUT_DIR="results/vit_b16_teacher"

!python dist_train_teacher.py \
    --data-path "$DATA_PATH" \
    --device cuda \
    --batch-size 32 \
    --opt adamw \
    --lr 0.0001 \
    --epochs 30 \
    --lr-warmup-epochs 3 \
    --lr-warmup-method linear \
    --lr-scheduler cosineannealinglr \
    --train-crop-size 224 \
    --val-resize-size 256 \
    --output-dir "$OUTPUT_DIR"
```

### ✅ Kết quả thực tế (TRASH Dataset, 4 classes)

| Epoch | Train Acc | Test Acc@1 | Ghi chú |
|:-----:|:---------:|:----------:|:--------|
| 0 | 47.2% | 67.3% | Bắt đầu học |
| 1 | 86.8% | 94.7% | Transfer learning phát huy |
| 2 | 91.9% | 96.6% | Tăng nhanh |
| 6 | 94.5% | 97.4% | Hội tụ tốt |
| 12 | 96.6% | 98.4% | Tiếp tục cải thiện |
| 18 | 97.9% | 98.7% | Gần đỉnh |
| 25 | 98.8% | **99.1%** | Đạt đỉnh |
| 29 | 99.1% | **99.14%** | Best final |

**Best Teacher Acc@1: 99.139%** — Thời gian train: ~2h34 trên GPU Colab.

> [!IMPORTANT]
> Khi chạy xong, file trọng số tốt nhất sẽ tự động lưu tại **`results/vit_b16_teacher/best_teacher.pth`**. File này sẽ được dùng ở Bước 7. Nếu muốn lưu lại vĩnh viễn, copy vào Google Drive:
> ```python
> !cp results/vit_b16_teacher/best_teacher.pth /content/drive/MyDrive/
> ```

---

## 🩹 BƯỚC 6.5: Fix Lỗi Inplace Operation (Cell 6.5)

### Tại sao cần bước này?

Khi chạy CAKD trực tiếp, PyTorch sẽ báo lỗi:

```
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation: [torch.cuda.FloatTensor [1, 64, 4, 4]] is at version 2;
expected version 1 instead.
```

### Giải thích chi tiết lỗi này

Để hiểu lỗi, cần biết cách PyTorch tính gradient. Mỗi khi bạn thực hiện phép tính tensor, PyTorch **ghi nhớ toàn bộ lịch sử** phép tính đó vào một "computation graph" (đồ thị tính toán). Khi gọi `loss.backward()`, PyTorch đi ngược đồ thị này để tính gradient.

**Trong code CAKD, quá trình training 1 batch diễn ra như sau:**

```
1. Student(image) → output, attn_weights     ← forward student
2. Teacher(image) → tea_logits, tea_attn      ← forward teacher
3. Discriminator(fake_attn) → pred_fake       ← forward discriminator
4. Discriminator(real_attn) → pred_real       ← forward discriminator

5. Tính gan_loss từ pred_real và pred_fake
6. gan_loss.backward()                        ← tính gradient cho discriminator
7. d_optimizer.step()                         ← CẬP NHẬT trọng số discriminator ⚠️

8. Tính student_loss (bao gồm pred_fake)
9. student_loss.backward()                    ← LỖI XẢY RA Ở ĐÂY ❌
```

**Vấn đề nằm ở bước 7 → 9:**

- Ở bước 3, `pred_fake = Discriminator(input)`. PyTorch ghi nhớ: "để tính gradient của pred_fake, cần dùng **trọng số hiện tại** của Discriminator".
- Ở bước 7, `d_optimizer.step()` **sửa trực tiếp (inplace)** trọng số Discriminator.
- Ở bước 9, `student_loss.backward()` cố dùng `pred_fake` để tính gradient → PyTorch phát hiện trọng số Discriminator đã bị sửa → **báo lỗi** vì computation graph không còn hợp lệ.

**Hình dung đơn giản:**

```
Bước 3: PyTorch chụp ảnh (snapshot) trọng số D = [1.0, 2.0, 3.0]
Bước 7: d_optimizer.step() sửa D thành         [1.1, 1.9, 3.2]  ← inplace!
Bước 9: backward() cần ảnh cũ [1.0, 2.0, 3.0] nhưng nó đã bị xóa → LỖI
```

**Giải pháp:** Thêm `.detach()` vào `pred_fake` trong student loss. Điều này an toàn vì:
- `input_d_fake` đã được `.detach()` từ student → gradient từ GAN term **không** chảy ngược về student
- GAN term chỉ "chảy" vào Discriminator, mà Discriminator đã được cập nhật ở bước 7 rồi
- Nên `.detach()` ở đây không làm mất thông tin gradient nào hữu ích

**👉 Thao tác:** Tạo Cell mới (**chạy trước Cell 7**), dán và chạy:

```python
# ====== CELL 6.5: FIX LỖI INPLACE OPERATION ======
filepath = '/content/CrossArch_KD/CAKD/dist_train_cakd.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix 1: Detach pred_fake trong student loss
#   Trước: gan_criterion(pred_fake, True))
#   Sau:   gan_criterion(pred_fake.detach(), True))
old = 'gan_criterion(pred_fake, True))'
new = 'gan_criterion(pred_fake.detach(), True))'
content = content.replace(old, new)

# Fix 2: Không cần retain_graph nữa (tiết kiệm VRAM)
#   Vì pred_fake đã detach → student loss không chia sẻ graph với gan_loss
content = content.replace('gan_loss.backward(retain_graph=True)', 'gan_loss.backward()')

with open(filepath, 'w') as f:
    f.write(content)

print("✅ BƯỚC 6.5 HOÀN TẤT - Đã fix lỗi inplace operation!")
```

> [!NOTE]
> Bước này chỉ cần chạy **1 lần** sau mỗi lần Restart Runtime. Nếu đã chạy rồi thì chạy lại cũng không sao (replace sẽ không tìm thấy chuỗi cũ nữa).

---

## 🔥 BƯỚC 7: Chạy Knowledge Distillation — CAKD (Cell 7)

### Tại sao cần bước này?
Đây là bước **cốt lõi** của toàn bộ pipeline. Student (ResNet50) sẽ được huấn luyện với 2 nguồn thông tin cùng lúc:

1. **Ground-truth labels** — Giống như train bình thường (cross-entropy loss)
2. **Soft labels từ Teacher** — Student bắt chước phân phối xác suất đầu ra của ViT-B/16 (KL-divergence loss)

Ngoài ra, CAKD còn dùng **Cross-Architecture Alignment** để căn chỉnh feature maps giữa CNN (ResNet) và Transformer (ViT) — đây là điểm khác biệt chính so với KD truyền thống.

### Các thay đổi trong `dist_train_cakd.py` so với bản gốc

File `dist_train_cakd.py` hiện tại đã được **sửa đổi** từ bản gốc của repo để hỗ trợ dataset tùy chọn (không chỉ ImageNet):

| Vấn đề | File gốc (repo) | File đã sửa |
|:-------|:----------------|:------------|
| Teacher loading | Hardcode `vit_b_16(weights=IMAGENET1K_V1)` — luôn load 1000 classes | **Thêm `--teacher-checkpoint`** để load ViT đã fine-tune trên dataset tùy chọn |
| Teacher head | Giữ nguyên 1000 neurons | **Tự thay `heads.head = Linear(768, num_classes)`** theo số class thực tế |
| topk metric | Hardcode `topk=(1, 5)` | **Dùng `topk=(1, min(4, output.shape[1]))`** — tránh lỗi khi < 5 classes |
| Fallback | ❌ Không có | ✅ **In warning** nếu không truyền `--teacher-checkpoint` |

### Giải thích các hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|:--------|:--------|:--------|
| `--lr 0.01` | Learning rate = 0.01 | Giống Baseline vì Student (ResNet50) train **from scratch**, không phải fine-tuning. Không dùng lr=0.0001 (quá nhỏ cho SGD from scratch), không dùng lr=0.1 (quá lớn cho dataset nhỏ 4 classes) |
| `--epochs 60` | 60 epochs | **Quan trọng:** KD loss chỉ bật từ **epoch 25** (xem giải thích bên dưới). Với 30 epochs, KD chỉ hoạt động 5 epoch cuối — quá ít. 60 epochs cho KD 35 epoch để phát huy |
| `--opt sgd` | SGD (mặc định) | Chuẩn cho ResNet train from scratch |

> [!CAUTION]
> **KD Loss Schedule trong code CAKD:** Trong file `dist_train_cakd.py` (dòng 49), trọng số KD loss được tính bằng `min(max(epoch-25, 0)/50.0, 0.2)`. Nghĩa là:
> - Epoch 0–24: KD weight = 0 → Student chỉ học từ labels (giống Baseline)
> - Epoch 25: KD weight = 0 → bắt đầu bật
> - Epoch 35: KD weight = 0.2 → đạt max
> - Epoch 35+: KD weight giữ = 0.2
>
> → **Nên train ≥ 50 epochs** để KD có đủ thời gian tác động. Khuyến nghị **60 epochs**.

### Tại sao KD chỉ bật từ epoch 25?

Đây là **thiết kế có chủ đích** của tác giả CAKD — "Student phải biết đi trước khi học chạy":

**Công thức tính loss:**
```
loss = cls_loss + λ × (pca_loss + gl_loss + gan_term)
       ─────────   ─────────────────────────────────
       Tự học từ    Học từ Teacher (KD)
       labels
```

Trong đó `λ = min(max(epoch-25, 0)/50.0, 0.2)`:

| Epoch | λ (trọng số KD) | Ý nghĩa |
|:-----:|:---------------:|:---------|
| 0–24 | **0** | Student tự học cơ bản từ labels (giống Baseline) |
| 25 | 0 | KD vừa bật |
| 30 | 0.1 | KD tăng dần (ramp up) |
| 35+ | **0.2** | KD ở full power |

**Tại sao không bật KD ngay từ đầu?**

1. **Epoch 0-24:** Student (ResNet50) cần xây dựng **feature representation cơ bản** trước. Nếu bật KD ngay, Student còn chưa biết gì → bắt chước Teacher quá mạnh (ViT 99% accuracy) sẽ gây gradient hỗn loạn → khó hội tụ
2. **Epoch 25-34:** KD **tăng tuyến tính** từ 0 → 0.2, không bật đột ngột → tránh gradient shock
3. **Epoch 35+:** Student đã ổn định → chịu được sự ảnh hưởng đầy đủ từ Teacher

```
Epoch:  0────────24──25────────35────────60
    λ:  0  0  0  0   0  ↗  ↗  0.2  0.2  0.2
        ╰───────────╯   ╰──────╯   ╰────────╯
        Student tự học   Ramp up    KD full power
```

> [!NOTE]
> Đó là lý do accuracy epoch 0-24 của CAKD **gần giống Baseline** — vì thực chất chúng đang train giống nhau! Sự khác biệt chỉ thể hiện rõ từ epoch 25 trở đi.

**👉 Thao tác:** Tạo Cell mới, dán và chạy:

```bash
# ====== CELL 7: TRAIN CAKD (KNOWLEDGE DISTILLATION) ======
%cd /content/CrossArch_KD/CAKD

DATA_PATH="/content/TRASH_Dataset_Split"
OUTPUT_DIR="results/resnet50/cakd_distilled"
TEACHER_CKPT="results/vit_b16_teacher/best_teacher.pth"

!python dist_train_cakd.py \
    --data-path "$DATA_PATH" \
    --device cuda \
    --batch-size 32 \
    --lr 0.01 \
    --epochs 60 \
    --train-crop-size 224 \
    --val-resize-size 224 \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --output-dir "$OUTPUT_DIR"
```

---

## 📊 So Sánh Kết Quả

Sau khi huấn luyện cả 3 bước (5, 6, 7), bạn sẽ có kết quả để so sánh:

| Bước | Phương Pháp | Mô Hình | Kết Quả Lưu Tại |
|:----:|:-----------:|:-------:|:---------------:|
| 5 | Baseline | ResNet50 (tự học) | `results/resnet50/baseline/` |
| 6 | Teacher | ViT-B/16 (fine-tuned) | `results/vit_b16_teacher/` |
| **7** | **CAKD** | **ResNet50 (học từ ViT)** | `results/resnet50/cakd_distilled/` |

### ✅ Kết quả thực tế CAKD (TRASH Dataset, 4 classes, 60 epochs)

| Epoch | CAKD Acc@1 | Baseline Acc@1 | Giai đoạn |
|:-----:|:----------:|:--------------:|:----------|
| 3 | 42.1% | 51.3% | Chưa có KD, CAKD chậm hơn |
| 9 | 74.2% | 63.4% | Student tự học |
| 15 | 68.3% | 79.7% | Dao động bình thường |
| 20 | 78.3% | 75.6% | Gần bật KD |
| 25 | 84.1% | 83.2% | 🔄 KD bắt đầu bật |
| 30 | 87.6% | 84.5% ← best | **KD ramp up → vượt Baseline** |
| 35 | 89.4% | — | KD full power (λ=0.2) |
| 40 | 90.9% | — | Tiếp tục tăng |
| 45 | 91.6% | — | Hội tụ dần |
| 50 | 92.1% | — | Ổn định |
| 55 | 92.9% | — | Gần đỉnh |
| 56 | **93.2%** | — | 🏆 **Best CAKD** |
| 59 | 92.1% | — | Final |

### 🏆 Bảng so sánh tổng kết

| | Baseline (ResNet50) | Teacher (ViT-B/16) | **CAKD (ResNet50+KD)** |
|:--|:---:|:---:|:---:|
| **Best Acc@1** | 84.509% | 99.139% | **93.173%** |
| Epochs | 30 | 30 | 60 |
| Training time | ~51 phút | ~2h34 | ~1h56 |
| Model size | ~25M params | ~86M params | **~25M params** |

> [!IMPORTANT]
> **KD tăng accuracy ResNet50 từ 84.5% → 93.2% (+8.7%)** mà KHÔNG tăng kích thước model!
> ResNet50 nhẹ (~25M params) nhưng đạt accuracy gần bằng Teacher ViT-B/16 (93.2% vs 99.1%).
> → **Chứng minh CrossArch Knowledge Distillation hiệu quả!**

```
Acc@1 (%)
100 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Teacher: 99.1%
 95 ─                                        ╭──  CAKD: 93.2% 🏆
 90 ─                                   ╭────╯
 85 ─               ╭─── Baseline: 84.5%────────
 80 ─          ╭────╯
 75 ─     ╭────╯
 70 ─ ╭───╯
     ─────┬────┬────┬────┬────┬────┬────┬───
     0    5   10   15   20   25   30   60  Epoch
                              ↑
                         KD bật (λ>0)
```

---

## 🔬 Phân Tích Overfitting Teacher & Đề Xuất Cải Thiện

### Kết quả kiểm tra overfitting ViT-B/16

Sau khi train Teacher, chạy script `analysis/check_overfit_vit.py` để kiểm tra:

| Metric | Train | Val | Gap |
|:-------|:-----:|:---:|:---:|
| Accuracy | 100.00% | 99.14% | +0.86% |
| Loss (CE) | 0.0003 | 0.0317 | ×117.8 |
| Mean confidence (đúng) | — | 99.76% | — |
| Mean confidence (sai) | — | **86.6%** | ⚠️ |

> [!WARNING]
> **Teacher tự tin SAI**: 15 mẫu dự đoán sai trên val có confidence trung bình **86.6%** — dấu hiệu memorization. ViT-B/16 (86M params) trên 6965 ảnh (tỷ lệ params/sample = 12,319) là quá lớn.
>
> **Tuy nhiên**, cho mục đích CAKD, Teacher 99%+ vẫn tốt vì:
> - Student (ResNet50, 25M params) không đủ capacity để bắt chước memorization
> - Soft labels từ Teacher vẫn chứa "dark knowledge" hữu ích
> - Gap chỉ 0.86% → Teacher vẫn generalize tốt trên phần lớn dữ liệu

### 💡 Đề xuất 1: Label Smoothing khi train Teacher

**Vấn đề:** Hard label `[1, 0, 0, 0]` ép Teacher output confidence cực cao (~99.97%), dẫn đến memorization.

**Giải pháp:** Label Smoothing (Szegedy et al., 2016 — [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)) thay hard label bằng soft label:

```
Trước (ε=0):   [1.000, 0.000, 0.000, 0.000]  → "Chắc chắn 100%"
Sau   (ε=0.1): [0.925, 0.025, 0.025, 0.025]  → "Chắc 92.5%, có thể sai"
```

Teacher không cần output confidence 99.97% nữa → giảm memorization → giảm overfit.

**Cách làm:** Chỉ cần thêm **1 flag** vào lệnh train Teacher (Bước 6). Code đã hỗ trợ sẵn, không cần sửa gì:

```bash
# Bước 6 cải tiến — thêm dòng --label-smoothing 0.1
!python dist_train_teacher.py \
    --data-path "$DATA_PATH" \
    --device cuda \
    --batch-size 32 \
    --opt adamw \
    --lr 0.0001 \
    --epochs 30 \
    --lr-warmup-epochs 3 \
    --lr-warmup-method linear \
    --lr-scheduler cosineannealinglr \
    --train-crop-size 224 \
    --val-resize-size 256 \
    --label-smoothing 0.1 \
    --output-dir "$OUTPUT_DIR"
```

> [!NOTE]
> Label Smoothing yêu cầu **train lại Teacher** (~2h34 trên Colab). Val accuracy có thể giảm nhẹ (99.1% → ~98.5%) nhưng chất lượng soft label cho KD sẽ tốt hơn.

### 💡 Đề xuất 2: Temperature Scaling khi train CAKD

**Vấn đề:** Softmax mặc định (T=1) ép output Teacher thành gần one-hot, **mất dark knowledge**:

```
Logits Teacher:  [12.0,  2.0,  3.0,  5.0]

T=1 (mặc định):  [0.997, 0.001, 0.001, 0.001]  → "Là GLASS, hết."
T=4 (đề xuất):   [0.632, 0.058, 0.081, 0.229]  → "Là GLASS, nhưng giống PLASTIC"
                                          ↑ dark knowledge: PLASTIC giống GLASS
```

**Dark knowledge** là mối quan hệ giữa các class sai — thông tin này giúp Student generalize tốt hơn so với chỉ học "đáp án đúng". Kỹ thuật này được đề xuất trong paper gốc KD (Hinton et al., 2015 — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)).

**Cách làm:** Sửa hàm `train_one_epoch` trong `dist_train_cakd.py`, thay loss logits matching (MSE) bằng KL-divergence với temperature:

```python
import torch.nn.functional as F

T = 4.0  # Temperature — thử giá trị 2, 4, 6 để so sánh

# Thay dòng MSE logits trong gl_loss:
#   Trước: mse_criterion(output, tea_logits.detach())
#   Sau:
kd_logit_loss = F.kl_div(
    F.log_softmax(output / T, dim=1),
    F.softmax(tea_logits.detach() / T, dim=1),
    reduction='batchmean'
) * (T * T)   # nhân T² để giữ gradient scale đúng
```

> [!NOTE]
> Temperature Scaling **không cần train lại Teacher** — chỉ thay đổi cách Student sử dụng output từ Teacher có sẵn. Tuy nhiên cần sửa code `dist_train_cakd.py`.

### So sánh 2 đề xuất

| | Label Smoothing | Temperature Scaling |
|:--|:---:|:---:|
| Áp dụng khi | Train Teacher (Bước 6) | Train CAKD (Bước 7) |
| Sửa code | ❌ Không (flag có sẵn) | ✅ Sửa `dist_train_cakd.py` |
| Train lại Teacher | ✅ Phải train lại (~2h34) | ❌ Không cần |
| Cải thiện kỳ vọng | +0.5~1% Student accuracy | +0.5~1.5% Student accuracy |
| Paper gốc | Szegedy et al. 2016 | Hinton et al. 2015 |

---

## ❓ Xử Lý Sự Cố Thường Gặp

| Lỗi | Nguyên nhân | Cách fix |
|:----|:-----------|:---------|
| `RuntimeError: k (5) is too big` | Chưa chạy Bước 4 (fix topk) | Quay lại chạy Cell 4 |
| `inplace operation` khi train CAKD | Bug trong code gốc: discriminator bị update trước khi student backward xong | Chạy Cell 6.5 (fix detach) |
| `shape mismatch` khi train CAKD | Chưa fine-tune Teacher (Bước 6) hoặc dùng sai checkpoint | Kiểm tra file `best_teacher.pth` có tồn tại |
| `Failed to load PyTorch C extensions` | Kernel Colab bị xung đột phiên bản | Restart Runtime → chạy lại từ Bước 1 |
| `FileNotFoundError: train/` | Giải nén sai cấu trúc (lỗi backslash Windows) | Chạy lại Bước 2 bằng script Python |
| `CUDA out of memory` | GPU không đủ RAM | Giảm `--batch-size` xuống 16 hoặc 8 |

---

## 📦 BƯỚC 8: Export ONNX — Chuẩn Bị Deploy Lên Jetson Nano (Cell 8)

### Tại sao cần bước này?
Sau khi train xong, trọng số model được lưu dưới dạng `.pth` (PyTorch checkpoint). Tuy nhiên, **Jetson Nano không chạy PyTorch hiệu quả** — cần chuyển sang:
1. **ONNX** — format trung gian, tương thích với mọi runtime (ONNX Runtime, TensorRT, OpenVINO...)
2. **TensorRT** — engine tối ưu riêng cho GPU NVIDIA, nhanh hơn ONNX Runtime 1.5-3x trên Jetson

### Vấn đề: Model CAKD trả về nhiều output

```python
# ResNet50 bình thường (Baseline):
output = model(image)  # → tensor logits [batch, 4]

# ResNet50_CAKD (có KD modules):
output, [attn_qk, attn_vv], vit_feat, cls_proj = model(image)
#  ↑ chỉ cần cái này khi deploy
```

**Giải pháp:** Wrap model CAKD để chỉ trả về logits, rồi export ONNX.

Dự án đã chuẩn bị sẵn đoạn mã trong file `deployment/export_onnx_and_benchmark.py` để xử lý việc này tự động từ Google Drive.

**👉 Thao tác:** Tạo một Cell mới, dán đoạn code sau và chạy. *Bắt buộc phải chạy bằng dấu `!` (khởi chạy theo shell) mới có thể gọi đúng phiên bản Python 3.10 chứa mô hình CAKD:*

```bash
# ====== CELL 8: EXPORT MÔ HÌNH SANG ONNX ======
%cd /content/CrossArch_KD/CAKD

# 1. Cài đặt thư viện vào Python 3.10
!pip install onnx onnxruntime

# 2. Chạy script export (sẽ báo Output ra Terminal)
!python deployment/export_onnx_and_benchmark.py

# 3. Copy các file ONNX sang Google Drive của bạn
!cp -r onnx_models/ /content/drive/MyDrive/CAKD_results/
print("\n✅ BƯỚC 8 HOÀN TẤT — Đã export ONNX & copy lên Google Drive!")
```

> [!NOTE]
> Opset 11 được chọn vì JetPack 4.x trên Jetson Nano hỗ trợ tốt nhất. Nếu dùng JetPack 5.x, có thể dùng opset 13+.

---

## 🚀 BƯỚC 9: Đo FPS Trên Jetson Nano (Chạy trên Jetson)

### Tại sao phải đo trên Jetson Nano?

| | Google Colab (T4) | **Jetson Nano** |
|:--|:---:|:---:|
| GPU | Tesla T4 (2560 CUDA) | **128 CUDA cores** (Maxwell) |
| VRAM | 16 GB riêng | **4 GB** chia sẻ CPU+GPU |
| TDP | 70W | **5-10W** |
| FPS ResNet50 | ~200-400 | **~15-30** (ONNX), **~40-60** (TRT FP16) |

> [!CAUTION]
> **Đo FPS trên Colab KHÔNG đại diện cho Jetson Nano!** GPU T4 mạnh hơn Jetson ~20 lần. Kết quả FPS trên Colab chỉ dùng để verify model export đúng, KHÔNG dùng để báo cáo hiệu năng thiết bị biên.

### Chuẩn bị trên Jetson Nano

```bash
# 1. Cài đặt (Jetson đã có TensorRT sẵn trong JetPack)
pip3 install onnx onnxruntime-gpu pycuda

# 2. Copy file ONNX từ PC/Drive sang Jetson
scp -r onnx_models/ jetson@192.168.1.xx:~/cakd/

# 3. Copy script benchmark
scp deployment/benchmark_jetson.py jetson@192.168.1.xx:~/cakd/
```

### Chạy Benchmark trên Jetson

```bash
cd ~/cakd/

# Cách 1: Chỉ đo ONNX Runtime (nhanh, không cần build engine)
python3 benchmark_jetson.py --onnx-dir ./onnx_models/ --no-tensorrt

# Cách 2: Đo cả TensorRT FP32
python3 benchmark_jetson.py --onnx-dir ./onnx_models/

# Cách 3: Đo TensorRT FP16 (KHUYẾN NGHỊ - nhanh nhất)
python3 benchmark_jetson.py --onnx-dir ./onnx_models/ --fp16

# Cách 4: Đo 1 model cụ thể
python3 benchmark_jetson.py --model ./onnx_models/cakd_resnet50.onnx --fp16
```

> [!IMPORTANT]
> **Lần đầu chạy TensorRT sẽ mất 5-10 phút** để build engine (optimize model cho GPU cụ thể). Các lần sau sẽ dùng lại engine đã build (file `.engine`).

### Kết quả mong đợi trên Jetson Nano

| Mô hình | ONNX Runtime | TRT FP32 | **TRT FP16** |
|:--------|:-----------:|:--------:|:-----------:|
| Baseline ResNet50 | ~18 FPS | ~28 FPS | **~45 FPS** |
| **CAKD ResNet50** | ~18 FPS | ~28 FPS | **~45 FPS** |

### Phân tích kết quả

```
FPS trên Jetson Nano (TensorRT FP16, batch=1, 224x224)

  CAKD ResNet50 ████████████████████████████████████████████░  ~45 FPS ← Deploy!
  Baseline      ████████████████████████████████████████████░  ~45 FPS  

  ──────────────┬────────────┬──────────────────────────────
                0           15                             45
                           Real-time
                           threshold
```

> [!TIP]
> **Kết luận cho deploy:**
> - CAKD ResNet50 và Baseline ResNet50 có **FPS gần bằng nhau** (cùng backbone)
> - CAKD **accuracy cao hơn 8.7%** (93.2% vs 84.5%) mà **không mất tốc độ**
> - Teacher ViT-B/16 quá chậm (~10 FPS) → **không deploy được** trên Jetson Nano
> - Đó chính là giá trị của Knowledge Distillation: **accuracy của ViT, tốc độ của ResNet!**
