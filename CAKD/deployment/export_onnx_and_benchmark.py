"""
==========================================================================
BƯỚC 8: EXPORT MÔ HÌNH SANG ONNX (Chạy trên Colab)
==========================================================================
Script này export 3 mô hình sang ONNX để chạy trên Jetson Nano.

Workflow:
  1. Chạy script này trên Colab (nơi có checkpoint .pth)
  2. Copy thư mục onnx_models/ lên Google Drive
  3. Từ Jetson Nano, tải file ONNX về
  4. Trên Jetson, dùng script benchmark_jetson.py để đo FPS

Lưu ý: Đo FPS trên Colab KHÔNG đại diện cho Jetson Nano vì:
  - Colab T4 có 2560 CUDA cores, Jetson Nano chỉ có 128
  - Colab có 16GB VRAM riêng, Jetson Nano chia sẻ 4GB RAM
  - Jetson dùng TensorRT tối ưu hơn nhiều so với ONNX Runtime
"""

import torch
import torch.nn as nn
import torchvision
import onnx
import numpy as np
import os


# ====================================================================
# WRAPPER CLASSES
# ====================================================================

class CAKDInferenceWrapper(nn.Module):
    """
    Wrapper cho ResNet50_CAKD - chỉ trả về logits.
    
    ResNet50_CAKD.forward() trả về:
        (output, [attn_qk, attn_vv], vit_feat, cls_proj)
    
    Khi deploy trên Jetson, chỉ cần 'output' (logits) để phân loại.
    Các output khác (attn, proj) chỉ dùng khi training KD.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output, _, _, _ = self.model(x)
        return output





# ====================================================================
# HÀM EXPORT ONNX
# ====================================================================

def export_to_onnx(model, onnx_path, input_size=(1, 3, 224, 224), 
                   device='cuda', opset_version=11):
    """
    Export PyTorch model sang ONNX.
    
    Sử dụng opset 11 (mặc định) vì Jetson Nano JetPack 4.x 
    hỗ trợ tốt opset 11. Nếu dùng JetPack 5.x có thể dùng opset 13+.
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # Test forward trước khi export
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"     Forward test OK - output shape: {test_output.shape}")
    
    # Export với dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Tối ưu constant folding
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Kiểm tra file ONNX hợp lệ
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"     ✅ Saved: {onnx_path} ({file_size_mb:.1f} MB)")
    
    return onnx_path


# ====================================================================
# MAIN
# ====================================================================

def main():
    NUM_CLASSES = 4  # Dataset TRASH: 4 classes
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'onnx_models'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 65)
    print("  BƯỚC 8: EXPORT MÔ HÌNH SANG ONNX (Baseline + CAKD)")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Output dir: {OUTPUT_DIR}/")
    print()
    
    exported = []
    
    # ------------------------------------------------------------------
    # 1. BASELINE RESNET50
    # ------------------------------------------------------------------
    print("📦 [1/2] Baseline ResNet50...")
    baseline_path_local = "results/resnet50/baseline/checkpoint.pth"
    baseline_path_gdrive = "/content/drive/MyDrive/CAKD_results/resnet50/baseline/checkpoint.pth"
    baseline_path = baseline_path_local if os.path.exists(baseline_path_local) else baseline_path_gdrive
    
    if os.path.exists(baseline_path):
        model = torchvision.models.resnet50(num_classes=NUM_CLASSES)
        ckpt = torch.load(baseline_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        
        onnx_path = os.path.join(OUTPUT_DIR, 'baseline_resnet50.onnx')
        export_to_onnx(model, onnx_path, device=DEVICE)
        exported.append(('Baseline ResNet50', onnx_path))
        del model, ckpt
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    else:
        print(f"  ⚠️  Không tìm thấy: {baseline_path}")
    
    # ------------------------------------------------------------------
    # 2. CAKD RESNET50
    # ------------------------------------------------------------------
    print("\n📦 [2/2] CAKD ResNet50 (Knowledge Distilled)...")
    cakd_path_local = "results/resnet50/cakd_distilled/checkpoint.pth"
    cakd_path_gdrive = "/content/drive/MyDrive/CAKD_results/resnet50/cakd_distilled/checkpoint.pth"
    cakd_path = cakd_path_local if os.path.exists(cakd_path_local) else cakd_path_gdrive
    
    if os.path.exists(cakd_path):
        cakd = torchvision.models.resnet50_cakd(num_classes=NUM_CLASSES)
        ckpt = torch.load(cakd_path, map_location='cpu')
        cakd.load_state_dict(ckpt['model'])
        wrapped = CAKDInferenceWrapper(cakd)
        
        onnx_path = os.path.join(OUTPUT_DIR, 'cakd_resnet50.onnx')
        export_to_onnx(wrapped, onnx_path, device=DEVICE)
        exported.append(('CAKD ResNet50', onnx_path))
        del cakd, wrapped, ckpt
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    else:
        print(f"  ⚠️  Không tìm thấy: {cakd_path}")
    
    # ------------------------------------------------------------------
    # TÓM TẮT
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  📊 KẾT QUẢ EXPORT")
    print("=" * 65)
    
    if exported:
        print(f"  {'Mô hình':<25} {'File ONNX':<35} {'Size':>8}")
        print(f"  {'-'*25} {'-'*35} {'-'*8}")
        for name, path in exported:
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name:<25} {path:<35} {size:>6.1f}MB")
    
    print(f"\n  📋 Bước tiếp theo trên Jetson Nano:")
    print(f"     1. Copy file ONNX sang Jetson")
    print(f"     2. Chạy: python3 benchmark_jetson.py")
    print(f"     3. Script sẽ tự convert TensorRT & đo FPS")
    print()


if __name__ == '__main__':
    main()
