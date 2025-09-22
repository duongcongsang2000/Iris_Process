import os, time, glob, threading, queue, argparse
from typing import Optional
import cv2
import numpy as np

# Giảm oversubscription CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# (tuỳ chọn) tránh TensorRT nếu bạn muốn dùng thuần CUDA
os.environ.setdefault("IRIS_DISABLE_TENSORRT", "1")

cv2.setNumThreads(1)

# Log providers của ORT
try:
    import onnxruntime as ort
    print("[BOOT] ORT providers (main):", ort.get_available_providers())
except Exception as e:
    print("[BOOT] ORT providers ERROR:", e)

# Open-Iris
from iris import IRISPipeline

_SENT = object()

def _bucket_number(name: str) -> Optional[int]:
    try:
        return int(name)
    except Exception:
        return None

def _out_path_for(src_path: str, input_dir: str, output_root: str) -> str:
    rel_path = os.path.relpath(src_path, input_dir)        # "014/L/xxx.jpg"
    rel_base, _ = os.path.splitext(rel_path)
    return os.path.join(output_root, "images", rel_base + ".png")

def list_images(input_dir: str, output_root: str,
                start_from_bucket: Optional[int], skip_existing: bool):
    files = []
    buckets = sorted([d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d))])
    for b in buckets:
        bn = _bucket_number(b)
        if start_from_bucket is not None and (bn is None or bn < start_from_bucket):
            continue
        for ext in ("jpg", "jpeg", "png"):
            cand = glob.glob(os.path.join(input_dir, b, f"**/*.{ext}"), recursive=True)
            if skip_existing:
                cand = [p for p in cand if not os.path.exists(_out_path_for(p, input_dir, output_root))]
            files.extend(cand)
    files.sort()
    return files

def load_worker(files, q: queue.Queue):
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        q.put((p, img), block=True)
    q.put(_SENT)

def save_worker(qsave: queue.Queue, total: int, log_every: int):
    saved = 0
    t0 = time.perf_counter()
    while True:
        item = qsave.get()
        if item is _SENT:
            break
        out_path, arr = item
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, arr)
        saved += 1
        if saved % log_every == 0 or saved == total:
            dt = time.perf_counter() - t0
            ips = saved / max(dt, 1e-9)
            print(f"[SAVE] {saved}/{total} | {ips:.2f} img/s")
    qsave.task_done()

def gpu_worker(qin: queue.Queue, qout: queue.Queue, input_dir: str, output_root: str, log_every: int):
    print("[GPU] init IRISPipeline()…")
    iris_pipeline = IRISPipeline()  # onnxruntime-gpu sẽ dùng CUDAExecutionProvider
    try:
        import onnxruntime as _ort
        print("[GPU] worker ORT providers:", _ort.get_available_providers())
    except Exception:
        pass
    print("[GPU] ready.")

    processed = 0
    t0 = time.perf_counter()

    while True:
        item = qin.get()
        if item is _SENT:
            qout.put(_SENT)
            break

        path_img, img = item
        if img is None:
            continue

        side_img = "left" if os.path.basename(os.path.dirname(path_img)) == "L" else "right"

        # Inference
        _ = iris_pipeline(img_data=img, eye_side=side_img)
        seg = iris_pipeline.call_trace['segmentation']
        seg_maps = seg.predictions
        if seg_maps is None or seg_maps.ndim != 3 or seg_maps.shape[2] < 3:
            continue

        mask_iris  = seg_maps[:, :, 1]
        mask_pupil = seg_maps[:, :, 2]
        mask = mask_iris - mask_pupil  # float32, cùng size HxW

        mmin, mmax = float(mask.min()), float(mask.max())
        if mmax <= mmin:
            continue

        # Chuẩn hoá về [0,1] và tạo phiên bản 8-bit để lưu nếu cần
        mask_norm = (mask - mmin) / (mmax - mmin)
        mask_u8   = (mask_norm * 255).astype(np.uint8)

        # Ảnh đã áp mask GIỮ NGUYÊN KÍCH THƯỚC GỐC
        iris_seg_full = (img.astype(np.float32) * mask_norm).astype(np.uint8)

        # Lưu ra đường dẫn tương ứng
        out_path_img  = _out_path_for(path_img, input_dir, output_root)  # .../images/...png
        # (tuỳ chọn) thêm thư mục masks song song
        out_path_mask = out_path_img.replace(os.sep + "images" + os.sep,
                                             os.sep + "masks"  + os.sep)

        # Đẩy sang thread saver
        qout.put((out_path_img,  iris_seg_full), block=True)
        # (tuỳ chọn) lưu luôn mask 8-bit cùng size
        qout.put((out_path_mask, mask_u8), block=True)

        processed += 1
        if processed % log_every == 0:
            dt = time.perf_counter() - t0
            ips = processed / max(dt, 1e-9)
            print(f"[GPU] processed: {processed} | {ips:.2f} img/s")
def parse_args():
    p = argparse.ArgumentParser(description="Open-IRIS GPU runner with resume")
    p.add_argument("--input",  default=os.path.join("CASIA-Iris-Lamp", ""),
                   help="Thư mục dataset gốc (default: Dataset/images)")
    p.add_argument("--output", default="CASIA-Iris-Lamp_Mask_2",
                   help="Thư mục output root (default: cropped)")
    p.add_argument("--start",  type=int, default=1,
                   help="Resume từ bucket số >= START (default: 14)")
    p.add_argument("--no-skip", action="store_true",
                   help="Không bỏ qua ảnh đã có output")
    p.add_argument("--io",     type=int, default=2,
                   help="Số thread đọc ảnh (default: 2)")
    p.add_argument("--qcap",   type=int, default=128,
                   help="Kích thước queue (default: 128)")
    p.add_argument("--log-every", type=int, default=100,
                   help="Log mỗi N ảnh (default: 100)")
    return p.parse_args()

def main():
    args = parse_args()

    input_dir   = args.input
    output_root = args.output
    start_from  = args.start
    skip_exist  = (not args.no_skip)
    q_cap       = max(8, args.qcap)
    io_threads  = max(1, args.io)
    log_every   = max(1, args.log_every)

    files = list_images(input_dir, output_root, start_from, skip_exist)
    total = len(files)
    if total == 0:
        print(f"[INFO] Không có ảnh cần xử lý trong {input_dir}. "
              f"start_from={start_from} | skip_existing={skip_exist}")
        return
    print(f"[INFO] Tìm thấy {total} ảnh cần xử lý | start_from>={start_from} | skip_existing={skip_exist}")

    chunks = [files[i::io_threads] for i in range(io_threads)]
    q_in  = queue.Queue(maxsize=q_cap)
    q_out = queue.Queue(maxsize=q_cap)

    loaders = []
    for part in chunks:
        t = threading.Thread(target=load_worker, args=(part, q_in), daemon=True)
        t.start()
        loaders.append(t)

    saver = threading.Thread(target=save_worker, args=(q_out, total, log_every), daemon=True)
    saver.start()

    gpu_worker(q_in, q_out, input_dir, output_root, log_every)

    for t in loaders:
        t.join()

    q_out.put(_SENT)
    saver.join()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
