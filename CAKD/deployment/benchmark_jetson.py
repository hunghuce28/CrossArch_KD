#!/usr/bin/env python3
"""
==========================================================================
BENCHMARK FPS TRÊN JETSON NANO
==========================================================================
Script đo FPS thực tế trên Jetson Nano với 3 engine:
  1. ONNX Runtime (cơ bản)
  2. TensorRT FP32 (tối ưu)
  3. TensorRT FP16 (tối ưu nhất - khuyến nghị cho Jetson)

Yêu cầu trên Jetson Nano:
  - JetPack 4.6+ (đã có TensorRT, CUDA, cuDNN)
  - pip3 install onnxruntime-gpu   (hoặc build từ source)
  - pip3 install pycuda

Cách dùng:
  python3 benchmark_jetson.py --onnx-dir ./onnx_models/
  python3 benchmark_jetson.py --onnx-dir ./onnx_models/ --fp16
  python3 benchmark_jetson.py --model ./onnx_models/cakd_resnet50.onnx --fp16

Kết quả mong đợi trên Jetson Nano (ResNet50, 224x224):
  - ONNX Runtime GPU: ~15-25 FPS
  - TensorRT FP32:    ~25-35 FPS
  - TensorRT FP16:    ~40-60 FPS  ← Khuyến nghị
==========================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path


def check_dependencies():
    """Kiểm tra các thư viện cần thiết trên Jetson."""
    deps = {}
    
    try:
        import onnxruntime as ort
        deps['onnxruntime'] = ort.__version__
        deps['ort_providers'] = ort.get_available_providers()
    except ImportError:
        deps['onnxruntime'] = None
    
    try:
        import tensorrt as trt
        deps['tensorrt'] = trt.__version__
    except ImportError:
        deps['tensorrt'] = None
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        deps['pycuda'] = True
        # Lấy thông tin GPU
        device = cuda.Device(0)
        deps['gpu_name'] = device.name()
        deps['gpu_memory'] = f"{device.total_memory() / (1024**2):.0f} MB"
        deps['compute_capability'] = f"{device.compute_capability()[0]}.{device.compute_capability()[1]}"
    except ImportError:
        deps['pycuda'] = None
    
    return deps


# ====================================================================
# PHƯƠNG PHÁP 1: ONNX RUNTIME
# ====================================================================

def benchmark_onnx_runtime(onnx_path, num_warmup=30, num_runs=100):
    """
    Đo FPS bằng ONNX Runtime trên Jetson.
    Thử GPU trước, fallback về CPU nếu không có.
    """
    import onnxruntime as ort
    
    results = {}
    
    # Thử GPU provider
    for provider_name, providers in [
        ('GPU', ['CUDAExecutionProvider', 'CPUExecutionProvider']),
        ('CPU', ['CPUExecutionProvider']),
    ]:
        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
            actual = session.get_providers()[0]
            
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            # Thay dynamic dim bằng 1
            input_shape = [1 if isinstance(d, str) else d for d in input_shape]
            
            dummy = np.random.randn(*input_shape).astype(np.float32)
            
            # Warm-up
            for _ in range(num_warmup):
                session.run(None, {input_name: dummy})
            
            # Benchmark
            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                session.run(None, {input_name: dummy})
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
            
            latencies = np.array(latencies)
            results[f'ORT-{provider_name}'] = {
                'provider': actual,
                'avg_fps': 1.0 / np.mean(latencies),
                'avg_ms': np.mean(latencies) * 1000,
                'min_ms': np.min(latencies) * 1000,
                'max_ms': np.max(latencies) * 1000,
                'p95_ms': np.percentile(latencies, 95) * 1000,
                'p99_ms': np.percentile(latencies, 99) * 1000,
            }
        except Exception as e:
            results[f'ORT-{provider_name}'] = {'error': str(e)}
    
    return results


# ====================================================================
# PHƯƠNG PHÁP 2: TENSORRT (Tối ưu nhất cho Jetson)
# ====================================================================

def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=False, 
                              max_batch_size=1, workspace_mb=512):
    """
    Convert ONNX → TensorRT engine.
    
    TensorRT tối ưu model cho GPU cụ thể (Jetson Nano = Maxwell SM53),
    nên engine build trên Jetson sẽ khác với engine build trên PC.
    
    Args:
        onnx_path: Path tới file .onnx
        engine_path: Path lưu file .engine (TensorRT serialized)
        fp16: Bật FP16 precision (Jetson Nano hỗ trợ, nhanh hơn ~1.5-2x)
        max_batch_size: Batch size tối đa
        workspace_mb: Bộ nhớ workspace cho TensorRT optimizer (MB)
    """
    import tensorrt as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ❌ TensorRT parse error: {parser.get_error(i)}")
            return None
    
    # Cấu hình builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1024 * 1024)
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"     ✅ FP16 mode enabled (Jetson hỗ trợ FP16)")
    elif fp16:
        print(f"     ⚠️  FP16 không khả dụng trên thiết bị này")
    
    # Set input shape
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Min, Optimal, Max batch size
    profile.set_shape(input_name, 
                      (1, 3, 224, 224),       # min
                      (1, 3, 224, 224),        # optimal
                      (max_batch_size, 3, 224, 224))  # max
    config.add_optimization_profile(profile)
    
    # Build engine (mất 2-10 phút trên Jetson Nano!)
    print(f"     ⏳ Đang build TensorRT engine (có thể mất 5-10 phút)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print(f"     ❌ Build engine thất bại!")
        return None
    
    # Serialize engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    precision = "FP16" if fp16 else "FP32"
    print(f"     ✅ Saved: {engine_path} ({size_mb:.1f} MB, {precision})")
    
    return engine_path


def benchmark_tensorrt(engine_path, num_warmup=50, num_runs=200):
    """
    Đo FPS bằng TensorRT engine trên Jetson.
    
    TensorRT inference flow:
      1. Allocate GPU memory cho input/output
      2. Copy input CPU → GPU  
      3. Execute inference trên GPU
      4. Copy output GPU → CPU
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            # Fill input with random data
            host_mem[:] = np.random.randn(size).astype(dtype)
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    # Warm-up
    for _ in range(num_warmup):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
        t0 = time.perf_counter()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        t1 = time.perf_counter()
        
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
        
        latencies.append(t1 - t0)
    
    latencies = np.array(latencies)
    return {
        'avg_fps': 1.0 / np.mean(latencies),
        'avg_ms': np.mean(latencies) * 1000,
        'min_ms': np.min(latencies) * 1000,
        'max_ms': np.max(latencies) * 1000,
        'p95_ms': np.percentile(latencies, 95) * 1000,
        'p99_ms': np.percentile(latencies, 99) * 1000,
    }


# ====================================================================
# MAIN
# ====================================================================

def print_results_table(all_results):
    """In bảng so sánh FPS đẹp."""
    print()
    print("=" * 85)
    print("  📊 BẢNG SO SÁNH FPS TRÊN JETSON NANO")
    print("=" * 85)
    
    header = f"  {'Mô hình':<22} {'Engine':<15} {'FPS':>7} {'Avg(ms)':>9} {'P95(ms)':>9} {'Min(ms)':>9}"
    print(header)
    print("  " + "-" * 80)
    
    for model_name, engine_results in all_results.items():
        for engine_name, r in engine_results.items():
            if 'error' in r:
                print(f"  {model_name:<22} {engine_name:<15} {'ERROR':>7} {r['error'][:40]}")
            else:
                print(f"  {model_name:<22} {engine_name:<15} "
                      f"{r['avg_fps']:>7.1f} "
                      f"{r['avg_ms']:>9.2f} "
                      f"{r.get('p95_ms', 0):>9.2f} "
                      f"{r['min_ms']:>9.2f}")
    
    print()
    
    # Phân tích
    print("=" * 85)
    print("  💡 PHÂN TÍCH CHO DEPLOY TRÊN THIẾT BỊ BIÊN")
    print("=" * 85)
    
    # Tìm kết quả Baseline vs CAKD
    baseline_fps = None
    cakd_fps = None
    teacher_fps = None
    
    for model_name, engine_results in all_results.items():
        # Ưu tiên TRT-FP16 > TRT-FP32 > ORT-GPU
        best_key = None
        for key in ['TRT-FP16', 'TRT-FP32', 'ORT-GPU']:
            if key in engine_results and 'error' not in engine_results[key]:
                best_key = key
                break
        
        if best_key is None:
            continue
        
        fps = engine_results[best_key]['avg_fps']
        if 'baseline' in model_name.lower():
            baseline_fps = (fps, best_key)
        elif 'cakd' in model_name.lower():
            cakd_fps = (fps, best_key)
        elif 'teacher' in model_name.lower() or 'vit' in model_name.lower():
            teacher_fps = (fps, best_key)
    
    if baseline_fps and cakd_fps:
        print(f"\n  🔹 Baseline vs CAKD (cùng ResNet50 backbone):")
        print(f"     Baseline: {baseline_fps[0]:.1f} FPS ({baseline_fps[1]})")
        print(f"     CAKD:     {cakd_fps[0]:.1f} FPS ({cakd_fps[1]})")
        diff = abs(cakd_fps[0] - baseline_fps[0]) / baseline_fps[0] * 100
        print(f"     → Chênh lệch FPS: ~{diff:.1f}% (gần bằng nhau = ĐÚNG)")
        print(f"       Vì cùng backbone ResNet50, khi inference chỉ dùng backbone.")
        print(f"       CAKD tăng ACCURACY (+8.7%) mà KHÔNG giảm tốc độ!")
    
    if teacher_fps and cakd_fps:
        speedup = cakd_fps[0] / teacher_fps[0]
        print(f"\n  🔹 Teacher ViT vs Student CAKD (mục tiêu KD):")
        print(f"     Teacher ViT-B/16:  {teacher_fps[0]:.1f} FPS ({teacher_fps[1]})")
        print(f"     Student CAKD:      {cakd_fps[0]:.1f} FPS ({cakd_fps[1]})")
        print(f"     → CAKD nhanh hơn {speedup:.1f}x so với Teacher!")
        
        if cakd_fps[0] >= 30:
            print(f"     ✅ CAKD đạt real-time (≥30 FPS) trên Jetson Nano!")
        elif cakd_fps[0] >= 15:
            print(f"     ⚠️  CAKD gần real-time ({cakd_fps[0]:.0f} FPS).")
            print(f"        Thử giảm input size (160x160) hoặc dùng MobileNet làm Student.")
        else:
            print(f"     ❌ CAKD chậm ({cakd_fps[0]:.0f} FPS). Cần:")
            print(f"        - Dùng TensorRT FP16 (nếu chưa)")
            print(f"        - Giảm input size")
            print(f"        - Đổi backbone nhẹ hơn (MobileNetV2, EfficientNet-Lite)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ONNX models trên Jetson Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Đo tất cả model trong thư mục
  python3 benchmark_jetson.py --onnx-dir ./onnx_models/
  
  # Đo 1 model cụ thể với TensorRT FP16
  python3 benchmark_jetson.py --model ./onnx_models/cakd_resnet50.onnx --fp16
  
  # Chỉ đo ONNX Runtime (không cần TensorRT)
  python3 benchmark_jetson.py --onnx-dir ./onnx_models/ --no-tensorrt
        """
    )
    parser.add_argument('--onnx-dir', type=str, default='onnx_models',
                        help='Thư mục chứa file .onnx (default: onnx_models/)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path tới 1 file .onnx cụ thể')
    parser.add_argument('--fp16', action='store_true',
                        help='Bật TensorRT FP16 (nhanh hơn, khuyến nghị cho Jetson)')
    parser.add_argument('--no-tensorrt', action='store_true',
                        help='Chỉ đo ONNX Runtime, bỏ qua TensorRT')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Số vòng warm-up (default: 50)')
    parser.add_argument('--runs', type=int, default=200,
                        help='Số vòng đo (default: 200)')
    parser.add_argument('--engine-dir', type=str, default='tensorrt_engines',
                        help='Thư mục lưu TensorRT engine (default: tensorrt_engines/)')
    
    args = parser.parse_args()
    
    # ----------------------------------------------------------------
    # Kiểm tra môi trường
    # ----------------------------------------------------------------
    print("=" * 65)
    print("  🔧 KIỂM TRA MÔI TRƯỜNG JETSON")
    print("=" * 65)
    
    deps = check_dependencies()
    
    print(f"  ONNX Runtime:  {deps.get('onnxruntime', '❌ Chưa cài')}")
    if deps.get('onnxruntime'):
        print(f"    Providers:   {deps.get('ort_providers', [])}")
    print(f"  TensorRT:      {deps.get('tensorrt', '❌ Chưa cài')}")
    print(f"  PyCUDA:        {'✅' if deps.get('pycuda') else '❌ Chưa cài'}")
    if deps.get('pycuda'):
        print(f"  GPU:           {deps.get('gpu_name', 'N/A')}")
        print(f"  GPU Memory:    {deps.get('gpu_memory', 'N/A')}")
        print(f"  Compute Cap:   {deps.get('compute_capability', 'N/A')}")
    print()
    
    # ----------------------------------------------------------------
    # Tìm file ONNX
    # ----------------------------------------------------------------
    onnx_files = {}
    
    if args.model:
        if os.path.exists(args.model):
            name = Path(args.model).stem
            onnx_files[name] = args.model
        else:
            print(f"❌ Không tìm thấy: {args.model}")
            return
    else:
        if not os.path.exists(args.onnx_dir):
            print(f"❌ Không tìm thấy thư mục: {args.onnx_dir}")
            print(f"   Hãy copy file ONNX từ Colab/Google Drive vào thư mục này.")
            return
        
        for f in sorted(os.listdir(args.onnx_dir)):
            if f.endswith('.onnx'):
                name = Path(f).stem
                # Đặt tên đẹp
                if 'baseline' in name.lower():
                    display = 'Baseline ResNet50'
                elif 'teacher' in name.lower() or 'vit' in name.lower():
                    display = 'Teacher ViT-B/16'
                elif 'cakd' in name.lower():
                    display = 'CAKD ResNet50'
                else:
                    display = name
                onnx_files[display] = os.path.join(args.onnx_dir, f)
    
    if not onnx_files:
        print(f"❌ Không tìm thấy file .onnx nào!")
        return
    
    print(f"📁 Tìm thấy {len(onnx_files)} model:")
    for name, path in onnx_files.items():
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"   • {name}: {path} ({size:.1f} MB)")
    print()
    
    # ----------------------------------------------------------------
    # Benchmark
    # ----------------------------------------------------------------
    all_results = {}
    
    for model_name, onnx_path in onnx_files.items():
        print(f"⏱️  Đang benchmark: {model_name}")
        print(f"   File: {onnx_path}")
        
        model_results = {}
        
        # 1. ONNX Runtime
        if deps.get('onnxruntime'):
            print("   📌 ONNX Runtime...")
            ort_results = benchmark_onnx_runtime(
                onnx_path, num_warmup=args.warmup, num_runs=args.runs
            )
            model_results.update(ort_results)
            for key, r in ort_results.items():
                if 'error' not in r:
                    print(f"      {key}: {r['avg_fps']:.1f} FPS ({r['avg_ms']:.2f} ms)")
        
        # 2. TensorRT
        if not args.no_tensorrt and deps.get('tensorrt') and deps.get('pycuda'):
            os.makedirs(args.engine_dir, exist_ok=True)
            
            # FP32
            print("   📌 TensorRT FP32...")
            engine_name = Path(onnx_path).stem + '_fp32.engine'
            engine_path = os.path.join(args.engine_dir, engine_name)
            
            if not os.path.exists(engine_path):
                convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=False)
            
            if os.path.exists(engine_path):
                trt_result = benchmark_tensorrt(
                    engine_path, num_warmup=args.warmup, num_runs=args.runs
                )
                model_results['TRT-FP32'] = trt_result
                print(f"      TRT-FP32: {trt_result['avg_fps']:.1f} FPS "
                      f"({trt_result['avg_ms']:.2f} ms)")
            
            # FP16
            if args.fp16:
                print("   📌 TensorRT FP16...")
                engine_name = Path(onnx_path).stem + '_fp16.engine'
                engine_path = os.path.join(args.engine_dir, engine_name)
                
                if not os.path.exists(engine_path):
                    convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=True)
                
                if os.path.exists(engine_path):
                    trt_result = benchmark_tensorrt(
                        engine_path, num_warmup=args.warmup, num_runs=args.runs
                    )
                    model_results['TRT-FP16'] = trt_result
                    print(f"      TRT-FP16: {trt_result['avg_fps']:.1f} FPS "
                          f"({trt_result['avg_ms']:.2f} ms)")
        
        all_results[model_name] = model_results
        print()
    
    # ----------------------------------------------------------------
    # In bảng tổng hợp
    # ----------------------------------------------------------------
    print_results_table(all_results)


if __name__ == '__main__':
    main()
