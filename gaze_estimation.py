"""
=============================================================
  GEOMETRIC GAZE ESTIMATION  -  Menu Launcher
=============================================================
  Chọn chế độ trong menu:
    1. Batch   - Xử lý ảnh từ input/, lưu kết quả ra output/ và data/
    2. Webcam  - Chạy real-time từ camera
    3. Vis 2D  - Vẽ tọa độ hình học lên ảnh gốc (debug)
    4. Vis 3D  - Mô hình hóa landmark trong hệ trục 3D (tương tác)
    0. Thoát
=============================================================
Phím tắt (chế độ Webcam):
    Q / ESC  Thoát    |  S  Lưu ảnh
    M  Bật/tắt Mesh   |  I  Bật/tắt ID landmark
=============================================================
"""

import cv2, sys, csv, time, threading
import numpy as np
from pathlib import Path
# Note: Heavy libraries (torch, mediapipe, ultralytics, plotly, pandas, filterpy, onnxruntime) are lazy-loaded unless specified.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------------------------------------------ #
#  CẤU HÌNH & NHẬN DIỆN PHẦN CỨNG                                     #
# ------------------------------------------------------------------ #
import platform
def is_raspberry_pi():
    try:
        m = Path('/proc/device-tree/model')
        return m.exists() and "Raspberry Pi" in m.read_text()
    except: return False

def get_hardware_info():
    arch = platform.machine().lower()
    is_rpi = is_raspberry_pi()
    return "Raspberry Pi 4" if is_rpi else f"Desktop ({arch.upper()})"

BASE_DIR        = Path(__file__).resolve().parent
YOLO_FACE_MODEL = BASE_DIR / "yolov8_head" / "yolov8n-face.pt"

# Folder setup
INPUT_DIR       = BASE_DIR / "input"
INPUT_VIDEO     = INPUT_DIR / "video"
OUTPUT_BATCH    = BASE_DIR / "output" / "face"
OUTPUT_VIS_2D   = BASE_DIR / "output" / "visualization"
OUTPUT_VIS_3D   = BASE_DIR / "output" / "visualization_html"
OUTPUT_WEBCAM   = BASE_DIR / "output" / "webcam"
OUTPUT_VIDEO    = BASE_DIR / "output" / "video"
DATA_DIR        = BASE_DIR / "data"  / "face"

for d in [INPUT_VIDEO, OUTPUT_BATCH, OUTPUT_VIS_2D, OUTPUT_VIS_3D, OUTPUT_WEBCAM, OUTPUT_VIDEO, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CAMERA_ID     = 0
GLOBAL_CONFIG = {
    "show_ids": False,
    "smooth_alpha": 0.4,
    "use_eye_gaze": True,
    "force_device": "cuda",
    "multi_face": True,
    "yolo_conf": 0.5,
    "roi_padding": 0.45,
    "detect_interval": 5,
    "high_fps_mode": False,
    "use_onnx": True,
    "hw_detected": get_hardware_info()
}

def auto_setup_hardware():
    """Tự động điều chỉnh cấu hình dựa trên phần cứng thực tế."""
    global GLOBAL_CONFIG
    arch = platform.machine().lower()
    is_rpi = is_raspberry_pi()
    
    if is_rpi or "arm" in arch or "aarch64" in arch:
        # Tối ưu cho Raspberry Pi / ARM
        GLOBAL_CONFIG["force_device"] = "cpu"
        GLOBAL_CONFIG["use_onnx"] = False      # Ưu tiên MediaPipe TFLite trên ARM
        GLOBAL_CONFIG["high_fps_mode"] = True   # Giảm tải tài nguyên
        GLOBAL_CONFIG["detect_interval"] = 8    # Giảm tần suất YOLO
        GLOBAL_CONFIG["multi_face"] = False     # Mặc định Fast mode cho mượt
        print(f"  [AUTO] Phát hiện Raspberry Pi/ARM. Đã cấu hình chế độ Low-Power.")
    else:
        # Tối ưu cho Desktop (Ưu tiên GPU nếu có)
        has_torch = False
        try:
            import torch
            has_torch = True
        except: pass
        # Deep Learning (Standard CPU versions)
        # Note: Torch on RPi is optional. The engine will fallback to ONNX/MediaPipe if missing.
        # If you need YOLO Native, uncomment below:
        # torch==2.5.1
        # torchvision==0.20.1
        if has_torch and torch.cuda.is_available():
            GLOBAL_CONFIG["force_device"] = "cuda"
            GLOBAL_CONFIG["use_onnx"] = True
            print(f"  [AUTO] Phát hiện Desktop + CUDA. Đã bật GPU Acceleration.")
        else:
            GLOBAL_CONFIG["force_device"] = "cpu"
            GLOBAL_CONFIG["use_onnx"] = False
            print(f"  [AUTO] Phát hiện Desktop CPU-only (hoặc thiếu Torch). Fallback về MediaPipe.")

# Gọi tự động khi khởi động
auto_setup_hardware()

# Màu sắc mặc định cho toàn bộ (Green - BGR)
FACE_COLOR = (0, 255, 0)

# ------------------------------------------------------------------ #
#  THREADED UTILITIES                                                #
# ------------------------------------------------------------------ #

class ThreadedCamera:
    """Luồng đọc camera độc lập để giữ frame luôn mới nhất."""
    def __init__(self, camera_id=0, width=1280, height=720):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_id, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Giới hạn buffer phần cứng
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

class ONNXGazeEngine:
    """
    Engine xử lý Inference sử dụng ONNX Runtime (GPU/CUDA).
    Hỗ trợ YOLOv8 Face Detection và FaceMesh Refined (478 pts).
    """
    def __init__(self):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        yolo_path = BASE_DIR / "c_ver" / "yolov8n-face.onnx"
        mesh_path = BASE_DIR / "c_ver" / "facemesh_refined.onnx"
        
        print(f"  [*] ONNX Engine: Đang khởi tạo trên {providers[0]}...")
        self.ort_yolo = ort.InferenceSession(str(yolo_path), providers=providers)
        self.ort_mesh = ort.InferenceSession(str(mesh_path), providers=providers)
        
        # Metadata
        self.yolo_input = self.ort_yolo.get_inputs()[0].name
        self.mesh_input = self.ort_mesh.get_inputs()[0].name
        self.mesh_outputs = [o.name for o in self.ort_mesh.get_outputs()]

    def detect(self, frame, conf_threshold=0.5):
        """Chạy YOLOv8 ONNX và trả về boxes định dạng tương đồng ultralytics."""
        h, w = frame.shape[:2]
        blob = cv2.resize(frame, (640, 640))
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        outputs = self.ort_yolo.run(None, {self.yolo_input: blob})
        # YOLOv8 output: [1, 5, 8400] -> [x, y, w, h, conf]
        out = outputs[0][0]
        
        boxes, confs = [], []
        for i in range(out.shape[1]):
            conf = out[4, i]
            if conf > conf_threshold:
                cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
                x1, y1 = (cx - bw/2) * (w/640), (cy - bh/2) * (h/640)
                x2, y2 = (cx + bw/2) * (w/640), (cy + bh/2) * (h/640)
                boxes.append([x1, y1, x2, y2])
                confs.append(float(conf))
        
        if not boxes: return []
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)
        
        # Giả lập object .xyxy tương đương ultralytics để reuse code
        class FakeBox:
            def __init__(self, x): self.xyxy = [np.array(x)]
            
        final_boxes = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            final_boxes.append(FakeBox(boxes[idx]))
        return final_boxes

    def process(self, rgb_crop):
        """Chạy FaceMesh Refined (478 pts) ONNX. Giả lập API của MediaPipe."""
        if rgb_crop is None or rgb_crop.size == 0: return None
        
        # Preprocess: 192x192 (rgb_crop đã là RGB từ process_frame)
        img = cv2.resize(rgb_crop, (192, 192))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # [1, 192, 192, 3]

        outputs = self.ort_mesh.run(None, {self.mesh_input: img})
        
        # Giải mã: refined model có 3 outputs: mesh(468), l_iris(5), r_iris(5)
        mesh_raw = outputs[0].reshape(-1, 3) # [468, 3]
        l_iris = outputs[1].reshape(-1, 2)   # [5, 2]
        r_iris = outputs[2].reshape(-1, 2)   # [5, 2]
        
        # Ghép thành 478 điểm
        full_lms = np.zeros((478, 3), dtype=np.float32)
        full_lms[:468] = mesh_raw
        
        # Map Iris (468-472: Left, 473-477: Right)
        for i in range(5):
            full_lms[468 + i] = [l_iris[i, 0], l_iris[i, 1], 0]
            full_lms[473 + i] = [r_iris[i, 0], r_iris[i, 1], 0]
            
        # Mock class tương đương mediapipe landmark object
        class Point:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
        class LandmarkList:
            def __init__(self, lms): self.landmark = [Point(p[0]/192, p[1]/192, p[2]/192) for p in lms]
        class Result:
            def __init__(self, ml, mw): self.multi_face_landmarks = [ml]; self.multi_face_world_landmarks = [mw]

        # Landmark pixel (scale 0-1 relative to 192x192)
        ml = LandmarkList(full_lms)
        # World landmark (Tận dụng tọa độ 3D của model)
        mw = LandmarkList(full_lms) 
        
        return Result(ml, mw)


def calculate_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1, area2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(area1 + area2 - inter + 1e-6)

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.eye(4, 7); self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.; self.kf.P *= 10.
        self.kf.Q[4:,4:] *= 0.01; self.kf.Q[-1,-1] *= 0.01
        self.kf.x[:4] = self._box_to_z(bbox)
        self.id, self.time_since_update, self.hits, self.hit_streak, self.age = KalmanBoxTracker.count, 0, 0, 0, 0
        self.history, self.smooth_gaze = [], None
        self.mesh = None # Lazy load per ID
        KalmanBoxTracker.count += 1

    def get_mesh(self):
        if self.mesh is None: self.mesh = make_face_mesh(static=False)
        return self.mesh

    def close(self):
        if self.mesh: self.mesh.close(); self.mesh = None

    def update(self, bbox):
        self.time_since_update, self.history = 0, []
        self.hits += 1; self.hit_streak += 1
        self.kf.update(self._box_to_z(bbox))

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <= 0: self.kf.x[6] *= 0.0
        self.kf.predict(); self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_box(self.kf.x)[0])
        return self.history[-1]

    def get_state(self): return self._x_to_box(self.kf.x)[0]

    def _box_to_z(self, b):
        w, h = b[2]-b[0], b[3]-b[1]
        return np.array([b[0]+w/2., b[1]+h/2., w*h, w/float(h)]).reshape((4,1))

    def _x_to_box(self, x):
        w = np.sqrt(x[2] * x[3]); h = x[2]/w
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))

class FaceTracker:
    def __init__(self, iou_threshold=0.3, max_lost=15, min_hits=3):
        from scipy.optimize import linear_sum_assignment
        self.iou_threshold, self.max_lost, self.min_hits = iou_threshold, max_lost, min_hits
        self.trackers, self.frame_count = [], 0

    def update(self, yolo_res):
        self.frame_count += 1
        dets = np.array([b.xyxy[0].tolist() for b in yolo_res]) if yolo_res else np.empty((0, 4))
        trks = np.array([t.predict() for t in self.trackers] or np.empty((0, 4)))
        
        # Build IOU matrix and Associate
        iou_mat = np.array([[calculate_iou(d, t) for t in trks] for d in dets]) if len(trks) else np.empty((len(dets), 0))
        matched, unmatched_dets, _ = self._associate(iou_mat, len(dets), len(trks))

        for m in matched: self.trackers[m[1]].update(dets[m[0]])
        for i in unmatched_dets: self.trackers.append(KalmanBoxTracker(dets[i]))
        
        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            if trk.time_since_update > self.max_lost: trk.close(); self.trackers.pop(i); continue
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Matching back to original YOLO objects for metadata
                best_idx = np.argmax([calculate_iou(trk.get_state(), d) for d in dets]) if len(dets) else -1
                if best_idx != -1 and calculate_iou(trk.get_state(), dets[best_idx]) > 0.5:
                    ret.append((yolo_res[best_idx], trk.id))
        return ret

    def _associate(self, iou_mat, num_dets, num_trks):
        from scipy.optimize import linear_sum_assignment
        if not num_trks or not num_dets: return np.empty((0,2), int), np.arange(num_dets), np.arange(num_trks)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched = np.array([[r, c] for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= self.iou_threshold])
        unmatched_dets = [i for i in range(num_dets) if i not in (matched[:,0] if len(matched) else [])]
        unmatched_trks = [i for i in range(num_trks) if i not in (matched[:,1] if len(matched) else [])]
        return matched, np.array(unmatched_dets), np.array(unmatched_trks)

    def get_tracker(self, tid):
        return next((t for t in self.trackers if t.id == tid), None)

    def get_smooth_gaze(self, tid):
        trk = self.get_tracker(tid)
        return trk.smooth_gaze if trk else None

    def set_smooth_gaze(self, tid, gaze_vec):
        trk = self.get_tracker(tid)
        if trk: trk.smooth_gaze = gaze_vec

def get_device():
    import torch
    return "cuda" if GLOBAL_CONFIG["force_device"] == "cuda" and torch.cuda.is_available() else "cpu"

def make_face_mesh(static=False):
    import mediapipe as mp
    _mp_fm = mp.solutions.face_mesh
    return _mp_fm.FaceMesh(
        static_image_mode      = static,
        max_num_faces          = 1,
        refine_landmarks       = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.60, # Tăng lên để bám landmark chặt chẽ hơn
    )

def preprocess_face(crop, min_dim=None):
    import cv2
    import numpy as np
    if crop is None or crop.size == 0: return crop, 0, 0
    
    # Lấy min_dim từ config nếu không truyền vào
    if min_dim is None:
        min_dim = 256 if GLOBAL_CONFIG.get("high_fps_mode", False) else 384

    orig_h, orig_w = crop.shape[:2]
    
    # 1. UPSCALE
    current_max = max(orig_h, orig_w)
    if current_max < min_dim:
        scale = min_dim / current_max
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        # INTER_NEAREST nhanh hơn INTER_CUBIC
        interp = cv2.INTER_NEAREST if GLOBAL_CONFIG.get("high_fps_mode", False) else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    
    # Nếu HIGH_FPS_MODE=True, bỏ qua CLAHE và Sharpen để cứu CPU
    if GLOBAL_CONFIG.get("high_fps_mode", False):
        return crop, orig_w, orig_h

    # 2. CLAHE
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    crop_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening (Làm nét nhẹ)
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    crop_proc = cv2.filter2D(crop_proc, -1, kernel)
    
    return crop_proc, orig_w, orig_h


def get_iris_connections():
    import mediapipe as mp
    return mp.solutions.face_mesh.FACEMESH_IRISES


KEY_IDS = [163, 157, 161, 154, 468, 390, 384, 388, 381, 473, 168]

# ------------------------------------------------------------------ #
#  HÌNH HỌC TÍNH TOÁN GAZE (BÁM SÁT 4 BƯỚC LÝ THUYẾT)                  #
# ------------------------------------------------------------------ #

def _pt(world, idx):
    import numpy as np
    return np.array(world[idx]) if idx in world else None

def step1_get_face_basis(p168, p2, p331, p102):
    import numpy as np
    """
    BƯỚC 1: HỆ TỌA ĐỘ KHUÔN MẶT
    Tạo bộ ba trực chuẩn [Vf (Nhìn thẳng), Rf (Ngang mặt), Uf (Dọc mặt)]
    """
    # 1. Trục Dọc Tham Chiếu (Từ Chóp mũi 2 -> Ấn đường 168)
    U_ref = p168 - p2
    U_ref /= (np.linalg.norm(U_ref) + 1e-9)
    
    # 2. Vector Nhìn Thẳng Cơ Sở (Vuông góc với mặt phẳng chứa 168, 331, 102)
    nf = np.cross(p331 - p168, p102 - p168)
    Vf = -nf / np.linalg.norm(nf) if np.linalg.norm(nf) > 1e-6 else np.array([0,0,-1.])
    if Vf[2] > 0: Vf = -Vf # Z hướng âm (về phía camera)
    
    # 3. Trục OX - Ngang Mặt (Vuông góc với Dọc tham chiếu và Hướng nhìn)
    Rf = np.cross(U_ref, Vf)
    Rf /= (np.linalg.norm(Rf) + 1e-9)
    
    # 4. Trục OY - Dọc Mặt (Vuông góc với Ngang mặt và Hướng nhìn)
    Uf = np.cross(Vf, Rf)
    Uf /= (np.linalg.norm(Uf) + 1e-9)
    
    return Vf, Rf, Uf

def step2_find_true_eyeball_center(P_top, P_bottom, P_inner, P_outer, V_face):
    import numpy as np
    """
    BƯỚC 2: TÂM NHÃN CẦU LÙI SÂU TRONG HỐC MẮT
    """
    # Tâm bề mặt khe hở
    O_surface = (P_top + P_bottom + P_inner + P_outer) / 4.0
    
    # Ước lượng bán kính nhãn cầu (tương đối theo khoảng cách 2 khoé ngoài-trong của mắt)
    # MediaPipe scale là chuẩn khoảng 3cm. 
    iris_radius_approx = np.linalg.norm(P_outer - P_inner) * 0.4
    
    # Lùi sâu (ngược hướng V_face, vì V_face đang hướng đâm về phía camera Z âm)
    # Điều này tạo một tâm khớp xoay ổn định.
    O_eyeball = O_surface - V_face * iris_radius_approx
    
    return O_eyeball

def calculate_gaze(world):
    import numpy as np
    """
    BƯỚC 3-4: RAYCAST TRỰC TIẾP QUA TÂM ĐỒNG TỬ 3D
    """
    use_eye = GLOBAL_CONFIG.get("use_eye_gaze", True)
    
    p = {
        'g': _pt(world, 168), 'n': _pt(world, 2), 
        'a': _pt(world, 331), 'b': _pt(world, 102)
    }
    if any(v is None for v in p.values()): return None, None, None, None

    # Lấy Hướng Mặt (cần dùng V_face để ước lượng chiều sâu)
    V_face, Rf, Uf = step1_get_face_basis(p['g'], p['n'], p['a'], p['b'])
    
    if not use_eye: 
        return None, None, p['g'], V_face

    # Mắt trái (Left: 163, 157, 161, 154 | Pupil: 468)
    eyeL = [_pt(world, i) for i in (163, 157, 161, 154, 468)]
    eyeR = [_pt(world, i) for i in (390, 384, 388, 381, 473)]
    
    if any(pt is None for pt in eyeL) or any(pt is None for pt in eyeR):
        return None, None, None, None

    # Mắt Trái
    O_L = step2_find_true_eyeball_center(eyeL[0], eyeL[1], eyeL[2], eyeL[3], V_face)
    gaze_vec_L = eyeL[4] - O_L
    nL = np.linalg.norm(gaze_vec_L)
    if nL > 1e-6: gaze_vec_L /= nL

    # Mắt Phải
    O_R = step2_find_true_eyeball_center(eyeR[0], eyeR[1], eyeR[2], eyeR[3], V_face)
    gaze_vec_R = eyeR[4] - O_R
    nR = np.linalg.norm(gaze_vec_R)
    if nR > 1e-6: gaze_vec_R /= nR

    # Tổng hợp tia Gaze mượt mà không bị lật góc (Wrap-around 180 deg)
    V_final = (gaze_vec_L + gaze_vec_R) / 2.0
    V_final /= np.linalg.norm(V_final)
    
    # Mô phỏng lại output để HUD hiển thị (Yaw / Pitch thay cho Alpha / Distance)
    # Yaw: hướng xoay trái phải. Pitch: ngước lên xuống. (Z âm hướng nhìn ta)
    pitch = np.degrees(np.arcsin(-V_final[1])) # Y của mp hướng xuống, nên lật lại để pitch âm là nhìn xuống
    yaw = np.degrees(np.arctan2(V_final[0], -V_final[2]))
    
    return yaw, pitch, p['g'], V_final


def build_coords(face_lms, world_lms, x1, y1, w_c, h_c):
    import numpy as np
    px, wc = {}, {}
    for i, lm in enumerate(face_lms.landmark):
        px[i] = (int(lm.x*w_c + x1), int(lm.y*h_c + y1))
    if world_lms:
        # MediaPipe World Landmarks (meters): Y+ is Down. Flip to Y-Up.
        for i, lm in enumerate(world_lms.landmark):
            wc[i] = np.array([lm.x, -lm.y, lm.z])
    else:
        # Fallback: Landmark chuẩn hóa (0-1). 
        # LƯU Ý: Y chuẩn hóa hướng XUỐNG, cần đảo ngược để Y hướng LÊN (geometry chuẩn).
        for i, lm in enumerate(face_lms.landmark):
            wc[i] = np.array([lm.x, -lm.y, lm.z]) # Đảo ngược Y
    return px, wc

# ------------------------------------------------------------------ #
#  VẼ                                                                  #
# ------------------------------------------------------------------ #
def draw_arrow(frame, px, V, w_c, h_c, color=(0,0,255)):
    """Vẽ tia hướng nhìn từ điểm 168 (-----°)."""
    if 168 not in px: return
    s = px[168]
    # Phóng đại Vx, Vy để vẽ hướng trên ảnh. Kéo dãn 0.6 lần (đã giảm xuống 30% từ 2.0)
    L = min(w_c, h_c) * 0.6
    # Vx+ (Right world) -> Right image. Vy+ (Up world) -> Up image (Giảm Y px).
    e = (int(s[0] + V[0]*L), int(s[1] - V[1]*L))

    
    # Thay mũi tên bằng đường line + chấm tròn ở cuối
    cv2.line(frame, s, e, color, 2)
    cv2.circle(frame, e, 5, color, -1) # Chấm ở đầu vector
    cv2.circle(frame, s, 3, color, -1) # Chấm ở gốc vector

def draw_iris(frame, px):
    for conn in get_iris_connections():
        s, e = conn
        if s in px and e in px:
            cv2.line(frame, px[s], px[e], (0,255,255), 1)
    for i in [468, 473]:
        if i in px:
            cv2.circle(frame, px[i], 3, (255,255,255), -1)

def draw_eye_geometry(frame, px, wc):
    """Vẽ tĩnh mạch/quỹ đạo nhãn cầu (chế độ Vis 2D)."""
    p168, p2_w, p331, p102 = _pt(wc, 168), _pt(wc, 2), _pt(wc, 331), _pt(wc, 102)
    if any(x is None for x in [p168, p2_w, p331, p102]): return
    V_face, _, _ = step1_get_face_basis(p168, p2_w, p331, p102)

    for ids in [(163,157,161,154,468), (390,384,388,381,473)]:
        p1,p2,p3,p4,_ = [_pt(wc, i) for i in ids]
        if any(x is None for x in [p1,p2,p3,p4]): continue
        
        O = step2_find_true_eyeball_center(p1,p2,p3,p4, V_face)

        if ids[0] in px and ids[2] in px:
            cv2.line(frame, px[ids[0]], px[ids[1]], (255,200,0), 2)
            cv2.line(frame, px[ids[2]], px[ids[3]], (255,200,0), 2)
        
        def _wc2px(pt3d, ref_idx, wc, px):
            if ref_idx not in wc or ref_idx not in px: return None
            ref_w = np.array(wc[ref_idx])
            ref_p = px[ref_idx]
            dp = pt3d - ref_w
            sc = 800
            return (int(ref_p[0] + dp[0]*sc), int(ref_p[1] + dp[1]*sc))
            
        origin_px = px.get(ids[0])
        if origin_px:
            op = _wc2px(O, ids[0], wc, px)
            if op:
                cv2.drawMarker(frame, op, (0,0,255), cv2.MARKER_CROSS, 8, 2)
                if ids[2] in px and ids[3] in px:
                    radius_2d = int(np.linalg.norm(np.array(px[ids[2]]) - np.array(px[ids[3]])) * 0.45)
                    cv2.circle(frame, op, radius_2d, (200, 0, 200), 1)

def draw_key_ids(frame, px):
    for i in KEY_IDS:
        if i in px:
            cv2.putText(frame, str(i), px[i], cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 1)

def draw_hud(frame, fps, yaw, pitch, show_mesh, show_ids):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (10,10), (330,135), (0,0,0), -1)
    cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
    st_fps = "HighPerf" if GLOBAL_CONFIG["high_fps_mode"] else "Normal"
    st_eng = "ONNX" if GLOBAL_CONFIG.get("use_onnx", True) else "M-Pipe"
    cv2.putText(frame, f"FPS: {fps:.1f} ({st_fps}|{st_eng})", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if st_fps=="Normal" else (255, 255, 0), 2)
    cv2.putText(frame, f"Yaw:   {yaw:.1f} deg" if yaw is not None else "Yaw:   --", (20,62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Pitch: {pitch:.1f} deg" if pitch is not None else "Pitch: --", (20,87), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Mesh:{'ON' if show_mesh else 'OFF'} IDs:{'ON' if show_ids else 'OFF'}", (20,112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, "Q/ESC Quit | S Save | M Mesh | I IDs", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

# ------------------------------------------------------------------ #
#  SHARED FRAME PROCESSOR (Batch + Webcam dùng chung)                  #
# ------------------------------------------------------------------ #
def process_frame(frame, tracked_faces, tracker=None, face_mesh=None, show_mesh=True, show_ids=False, vis2d=False):
    oh, ow = frame.shape[:2]
    rows = []
    for box, tid in (tracked_faces or []):
        # Mặc định lấy box thô từ YOLO
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Use per-ID dedicated FaceMesh if tracker exists, else fallback to face_mesh
        trk = tracker.get_tracker(tid) if tracker else None
        
        # Nếu đang tracking, ƯU TIÊN lấy box đã được làm mượt (smoothed) bởi Kalman Filter
        # Việc này giúp khuôn cắt (crop) không bị nháy, từ đó MediaPipe Face Mesh ổn định hơn rất nhiều.
        if trk:
            sx1, sy1, sx2, sy2 = map(int, trk.get_state())
            # Chỉ ghi đè nếu tọa độ hộp hợp lệ (không bị vượt biên quá lố)
            if sx2 > sx1 and sy2 > sy1:
                x1, y1, x2, y2 = max(0, sx1), max(0, sy1), min(ow, sx2), min(oh, sy2)
                
        # Padded Crop for Face Mesh (Direct ROI)
        bw, bh = x2-x1, y2-y1
        p_val = GLOBAL_CONFIG.get("roi_padding", 0.355)
        pad_w, pad_h = int(bw*p_val), int(bh*p_val)
        cx1, cy1 = max(0, x1-pad_w), max(0, y1-pad_h)
        cx2, cy2 = min(ow, x2+pad_w), min(oh, y2+pad_h)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0: continue
        
        # Apply Accuracy Enhancements (CLAHE + Sharpening)
        crop, orw, orh = preprocess_face(crop)
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), FACE_COLOR, 1)

        # Use per-ID dedicated FaceMesh if tracker exists, else fallback to face_mesh
        trk = tracker.get_tracker(tid) if tracker else None
        mesh_obj = (trk.get_mesh() if trk else None) or face_mesh
        if not mesh_obj: continue

        mpr = mesh_obj.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if not mpr.multi_face_landmarks: continue
        px, wc_d = build_coords(mpr.multi_face_landmarks[0], getattr(mpr, 'multi_face_world_landmarks', [None])[0], cx1, cy1, orw, orh)

        if vis2d: (draw_iris(frame, px), draw_eye_geometry(frame, px, wc_d))
        if show_ids or vis2d: draw_key_ids(frame, px)

        yaw, pitch, _, Vf = calculate_gaze(wc_d)
        if Vf is not None:
            if tracker:
                prev_v = tracker.get_smooth_gaze(tid)
                if prev_v is not None:
                    smooth = GLOBAL_CONFIG.get("smooth_alpha", 0.3)
                    Vf = (smooth*Vf + (1-smooth)*prev_v); Vf /= np.linalg.norm(Vf)
                tracker.set_smooth_gaze(tid, Vf)
            
            draw_arrow(frame, px, Vf, crop.shape[1], crop.shape[0], color=FACE_COLOR)
            txt = f"Y:{yaw:.1f} P:{pitch:.1f}" if yaw is not None else "FACE MODE"
            cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FACE_COLOR, 1)
            rows.append({"yaw_deg": yaw or 0, "pitch_deg": pitch or 0, "final_gaze_vector": f"({Vf[0]:.4f}, {Vf[1]:.4f}, {Vf[2]:.4f})"})
    return rows, None

# ------------------------------------------------------------------ #
#  CHẾ ĐỘ 1 - BATCH                                                   #
# ------------------------------------------------------------------ #
def run_batch():
    from ultralytics import YOLO
    print("\n[MODE] Batch Processing")
    yolo = YOLO(str(YOLO_FACE_MODEL)).to(get_device())
    fm   = make_face_mesh(static=True)
    imgs = list(INPUT_DIR.glob("*.[jJpP][pPnN][gG]*"))
    print(f"  Tìm thấy {len(imgs)} ảnh.")

    all_rows = []
    for i, path in enumerate(imgs):
        print(f"  [{i+1}/{len(imgs)}] Đang xử lý: {path.name}...")
        frame = cv2.imread(str(path))
        if frame is None: continue
        res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
        tracked = [(box, i) for i, box in enumerate(res.boxes)] if res.boxes else []
        rows, _ = process_frame(frame, tracked, face_mesh=fm, show_mesh=True, show_ids=GLOBAL_CONFIG["show_ids"])
        for r in rows:
            all_rows.append({"image_name": path.name, **r})
        cv2.imwrite(str(OUTPUT_BATCH/path.name), frame)
        print(f"    [✓] Hoàn tất {path.name}")

    if all_rows:
        csvp = DATA_DIR / "gaze_results.csv"
        with open(csvp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader(); w.writerows(all_rows)
        print(f"  Đã lưu CSV: {csvp}")
    fm.close()
    print("[DONE] Batch hoàn thành.\n")

# ------------------------------------------------------------------ #
#  CHẾ ĐỘ 2 - WEBCAM (THREADED CAMERA)                                #
# ------------------------------------------------------------------ #
def run_webcam():
    print("\n[MODE] Webcam Real-time")
    dev = get_device()
    use_onnx = GLOBAL_CONFIG.get("use_onnx", True)
    
    # 1. Khởi tạo Engine
    onnx_engine = None
    yolo_model = None
    if use_onnx:
        try:
            onnx_engine = ONNXGazeEngine()
            print("  [✓] Đã khởi tạo ONNX GPU Engine.")
        except Exception as e:
            print(f"  [!] Lỗi khởi tạo ONNX: {e}. Fallback sang MediaPipe.")
            use_onnx = False

    if not use_onnx:
        from ultralytics import YOLO
        print(f"  [*] Khởi tạo MediaPipe + YOLO trên: {dev.upper()}")
        yolo_model = YOLO(str(YOLO_FACE_MODEL)).to(dev)

    # 2. Khởi tạo Camera
    cam = ThreadedCamera(CAMERA_ID)
    if not cam.cap.isOpened():
        print(f"  [ERR] Không mở được camera ID {CAMERA_ID}."); return

    print("  [*] Đang khởi động camera...")
    for _ in range(15):
        grabbed, _ = cam.read()
        if grabbed: break
        time.sleep(0.1)
    if not grabbed:
        print("  [ERR] Camera không truyền được dữ liệu."); cam.stop(); return

    cam.start()
    print("  [*] Webcam đã sẵn sàng.")

    show_mesh, show_ids = True, GLOBAL_CONFIG["show_ids"]
    prev_t, save_n = time.time(), 0
    last_a, last_d = None, None

    print("  [*] Bắt đầu hiển thị. Nhấn 'Q' để thoát.")
    tracker = FaceTracker()
    last_tracked_faces = []
    frame_idx = 0
    fm_shared = None

    try:
        while True:
            grabbed, frame = cam.read()
            if not grabbed or frame is None: break
            frame_idx += 1

            t_start = time.time()
            multi = GLOBAL_CONFIG["multi_face"]
            t_yolo = 0
            
            # 3. Detection & Inference Pipeline
            if multi:
                interval = GLOBAL_CONFIG.get("detect_interval", 3)
                if frame_idx % interval == 0 or not last_tracked_faces:
                    t0 = time.time()
                    if use_onnx:
                        boxes = onnx_engine.detect(frame, conf_threshold=GLOBAL_CONFIG["yolo_conf"])
                    else:
                        res = yolo_model(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
                        boxes = res.boxes
                    last_tracked_faces = tracker.update(boxes)
                    t_yolo = (time.time() - t0) * 1000
                
                # Truyền onnx_engine vào process_frame nếu cần
                rows, _ = process_frame(frame, last_tracked_faces, tracker=tracker, 
                                        face_mesh=onnx_engine if use_onnx else None, 
                                        show_mesh=show_mesh, show_ids=show_ids)
            else:
                # Fast Mode (Single Face)
                if not use_onnx and not fm_shared: fm_shared = make_face_mesh(static=False)
                t0 = time.time()
                if use_onnx:
                    boxes = onnx_engine.detect(frame, conf_threshold=GLOBAL_CONFIG["yolo_conf"])
                else:
                    res = yolo_model(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=320, verbose=False)[0]
                    boxes = res.boxes
                t_yolo = (time.time() - t0) * 1000
                
                if boxes:
                    # Lấy mặt to nhất
                    if use_onnx:
                        # boxes ở đây là list FakeBox
                        best_box = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                    else:
                        best_box = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                    
                    rows, _ = process_frame(frame, [(best_box, 0)], 
                                            face_mesh=onnx_engine if use_onnx else fm_shared, 
                                            show_mesh=show_mesh, show_ids=show_ids)
                else: rows = []

            if rows:
                last_a, last_d = rows[-1]['yaw_deg'], rows[-1]['pitch_deg']

            t_total = (time.time() - t_start) * 1000
            fps = 1/(time.time()-prev_t+1e-9); prev_t = time.time()
            
            # Print profiling every 30 frames
            if frame_idx % 30 == 0:
                print(f"  [PROFILE] Total: {t_total:.1f}ms | YOLO: {t_yolo:.1f}ms | FPS: {fps:.1f}")

            draw_hud(frame, fps, last_a, last_d, show_mesh, show_ids)
            cv2.imshow("Gaze Estimation - Webcam", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                print("  [*] Thoát theo yêu cầu (Q/ESC).")
                break
            elif k == ord('s'):
                save_n += 1
                p = OUTPUT_WEBCAM / f"cap_{save_n:04d}.png"
                cv2.imwrite(str(p), frame); print(f"  [✓] Lưu: {p.name}")
            elif k == ord('m'): show_mesh = not show_mesh
            elif k == ord('i'): show_ids  = not show_ids

            if cv2.getWindowProperty("Gaze Estimation - Webcam", cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"  [ERR] Lỗi khi xử lý frame: {e}")
        import traceback; traceback.print_exc()
    finally:
        if fm_shared: fm_shared.close()
        cam.stop()
        cv2.destroyAllWindows()
    print("[DONE] Webcam đã đóng.\n")

# ------------------------------------------------------------------ #
#  CHẾ ĐỘ 3 - VIS 2D                                                  #
# ------------------------------------------------------------------ #
def run_vis2d():
    from ultralytics import YOLO
    print("\n[MODE] 2D Visualization")
    dev  = get_device()
    yolo = YOLO(str(YOLO_FACE_MODEL)).to(dev)
    fm   = make_face_mesh(static=True)
    imgs = list(INPUT_DIR.glob("*.[jJpP][pPnN][gG]*"))
    print(f"  Tìm thấy {len(imgs)} ảnh.")

    for i, path in enumerate(imgs):
        print(f"  [{i+1}/{len(imgs)}] Đang vẽ: {path.name}...")
        frame = cv2.imread(str(path))
        if frame is None: continue
        res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
        tracked = [(box, i) for i, box in enumerate(res.boxes)] if res.boxes else []
        process_frame(frame, tracked, face_mesh=fm, show_mesh=True, show_ids=GLOBAL_CONFIG["show_ids"], vis2d=True)
        out = OUTPUT_VIS_2D / f"vis_{path.name}"
        cv2.imwrite(str(out), frame)
        print(f"    [✓] {out.name}")
    fm.close()
    print(f"[DONE] Ảnh đã lưu tại {OUTPUT_VIS_2D}\n")

# ------------------------------------------------------------------ #
#  CHẾ ĐỘ 4 - VIS 3D                                                  #
# ------------------------------------------------------------------ #
def run_vis3d():
    from ultralytics import YOLO
    import plotly.graph_objects as go
    import numpy as np
    print("\n[MODE] 3D HTML Visualization (Plotly)")
    dev  = get_device()
    yolo = YOLO(str(YOLO_FACE_MODEL)).to(dev)
    fm   = make_face_mesh(static=True)
    imgs = list(INPUT_DIR.glob("*.[jJpP][pPnN][gG]*"))
    print(f"  Tìm thấy {len(imgs)} ảnh.")

    for i, path in enumerate(imgs):
        print(f"  [{i+1}/{len(imgs)}] Đang xử lý: {path.name}...")
        try:
            frame = cv2.imread(str(path))
            if frame is None: continue
            oh, ow = frame.shape[:2]
            res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
            if not res.boxes: continue
            box = res.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bw, bh = x2-x1, y2-y1
            p_val = GLOBAL_CONFIG.get("roi_padding", 0.355)
            pad_w, pad_h = int(bw*p_val), int(bh*p_val)
            cx1, cy1 = max(0, x1-pad_w), max(0, y1-pad_h)
            cx2, cy2 = min(ow, x2+pad_w), min(oh, y2+pad_h)
            crop = frame[cy1:cy2, cx1:cx2]
            crop, orw, orh = preprocess_face(crop)
            mpr = fm.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not mpr.multi_face_landmarks: continue

            wlms = getattr(mpr, 'multi_face_world_landmarks', None)
            wlms0 = wlms[0] if wlms else None
            _, wc_d = build_coords(mpr.multi_face_landmarks[0], wlms0, cx1, cy1, orw, orh)

            fig = go.Figure()

            # 1. Landmark mặt (vùng trên)
            log_data = [f"=== GAZE 3D CALC LOG: {path.name} ==="]
            
            # Pre-retrieve basis and final gaze to avoid repeats
            p168_wc = _pt(wc_d, 168)
            p2_wc = _pt(wc_d, 2)
            p331_wc = _pt(wc_d, 331)
            p102_wc = _pt(wc_d, 102)
            
            p_basis = all(p is not None for p in [p168_wc, p2_wc, p331_wc, p102_wc])
            V_face = None
            if p_basis:
                V_face, _, _ = step1_get_face_basis(p168_wc, p2_wc, p331_wc, p102_wc)
            
            yaw, pitch, _, Vf = calculate_gaze(wc_d)
            st_yaw = f"{yaw:.1f}" if yaw is not None else "--"
            st_pitch = f"{pitch:.1f}" if pitch is not None else "--"

            # 2. Các thành phần hố mắt & Iris (Nhóm 2)
            for eye_idx, eye_ids in enumerate([(163,157,161,154,468),(390,384,388,381,473)]):
                ep = [_pt(wc_d, idx) for idx in eye_ids[:4]]
                if any(p is None for p in ep): continue
                p1, p2, p3, p4 = ep # 163/390(outer), 157/384(top), 161/388(inner), 154/381(bottom)
                
                # Kiểm tra độ mở của mắt (Eye Openness)
                eye_open_dist = np.linalg.norm(p2 - p4)
                closed_thresh = 0.0055 
                is_closed = eye_open_dist < closed_thresh
                eye_label = "LEFT" if eye_idx == 0 else "RIGHT"
                
                # L1, L2 (Cyan cho mắt trái, Lime cho mắt phải)
                color_eye = 'cyan' if eye_idx == 0 else 'lime'
                # Vẽ Hố mắt (4 điểm chính)
                fig.add_trace(go.Scatter3d(x=[p1[0], p3[0]], y=[p1[1], p3[1]], z=[p1[2], p3[2]], mode='lines', line=dict(color=color_eye, width=3), name=f'Trục hốc mắt {eye_label} [2]', meta='2', showlegend=(eye_idx==0)))
                fig.add_trace(go.Scatter3d(x=[p2[0], p4[0]], y=[p2[1], p4[1]], z=[p2[2], p4[2]], mode='lines', line=dict(color=color_eye, width=3), meta='2', showlegend=False))
                
                # VẼ VIỀN MẮT TOÀN BỘ (Khóe mắt - Nhóm 2)
                import mediapipe as mp
                eye_conns = mp.solutions.face_mesh.FACEMESH_LEFT_EYE if eye_idx == 0 else mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
                lx, ly, lz = [], [], []
                for s, e in eye_conns:
                    if s in wc_d and e in wc_d:
                        lx.extend([wc_d[s][0], wc_d[e][0], None])
                        ly.extend([wc_d[s][1], wc_d[e][1], None])
                        lz.extend([wc_d[s][2], wc_d[e][2], None])
                if lx:
                    fig.add_trace(go.Scatter3d(
                        x=lx, y=ly, z=lz, 
                        mode='lines', line=dict(color=color_eye, width=2), 
                        name=f'Khóe mắt {eye_label} [2]', meta='2', showlegend=False
                    ))
                
                # VẼ IRIS 3D (Nếu có)
                for conn in get_iris_connections():
                    s, e = conn
                    if s in wc_d and e in wc_d:
                        ps, pe = wc_d[s], wc_d[e]
                        fig.add_trace(go.Scatter3d(
                            x=[ps[0], pe[0]], y=[ps[1], pe[1]], z=[ps[2], pe[2]], 
                            mode='lines', line=dict(color='yellow', width=2), 
                            opacity=0.8, meta='2', showlegend=False
                        ))

                if V_face is not None:
                    O = step2_find_true_eyeball_center(p2, p4, p3, p1, V_face) # Order: Top, Bottom, Inner, Outer
                    
                    log_data.append(f"\n[{eye_label} Eye] | OpenDist: {eye_open_dist:.5f} {'(CLOSED)' if is_closed else '(OPEN)'}")
                    log_data.append(f"  Eyeball Center (O): {O}")
                    
                    fig.add_trace(go.Scatter3d(
                        x=[O[0]], y=[O[1]], z=[O[2]], 
                        mode='markers', marker=dict(size=4, symbol='diamond', color='magenta', opacity=0.6), 
                        name=f'Tâm Nhãn Cầu {eye_label} [3]' if eye_idx==0 else "", meta='3',
                        showlegend=eye_idx==0
                    ))

                    # Pupil
                    pupil = _pt(wc_d, eye_ids[4])
                    if pupil is not None:
                        # Thêm điểm đồng tử với ID
                        fig.add_trace(go.Scatter3d(
                            x=[pupil[0]], y=[pupil[1]], z=[pupil[2]], 
                            mode='markers+text', marker=dict(size=5, color='darkblue'), 
                            text=[str(eye_ids[4])], textposition="bottom center",
                            name=f'Đồng tử {eye_label} [2]', meta='2', showlegend=False
                        ))
                        
                        # Ray from Center through pupil
                        gaze = (pupil - O)
                        g_norm = np.linalg.norm(gaze)
                        gaze /= (g_norm + 1e-9)
                        
                        # Màu tia nhìn: Xám nếu nhắm, màu đậm nếu mở
                        ray_color = 'gray' if is_closed else ('deepskyblue' if eye_idx==0 else 'lawngreen')
                        ray_dash = 'dash' if is_closed else 'solid'
                        ray_name = f'Tia {eye_label} {"(Nhắm)" if is_closed else ""} [3]'
                        
                        proj_pt = O + gaze * 0.05
                        fig.add_trace(go.Scatter3d(
                            x=[O[0], proj_pt[0]], y=[O[1], proj_pt[1]], z=[O[2], proj_pt[2]], 
                            mode='lines', line=dict(color=ray_color, width=4, dash=ray_dash), 
                            name=ray_name, meta='3'
                        ))

            # 3. Face Basis & Final Gaze (Nhóm 3)
            if p_basis:
                nf, rf, uf = step1_get_face_basis(p168_wc, p2_wc, p331_wc, p102_wc)
                s_f = 0.08
                for v, c, n in [(nf, 'black', 'Pháp tuyến (Front) [3]'), (rf, 'orange', 'Ngang mặt (Right) [3]'), (uf, 'purple', 'Dọc mặt (Up) [3]')]:
                    fig.add_trace(go.Scatter3d(x=[p168_wc[0], p168_wc[0]+v[0]*s_f], y=[p168_wc[1], p168_wc[1]+v[1]*s_f], z=[p168_wc[2], p168_wc[2]+v[2]*s_f], mode='lines', line=dict(color=c, width=6), name=n, meta='3'))
                
                if Vf is not None:
                    # Final Gaze from Glabella (168) - Reduced to 30% (from 0.18 to 0.054)
                    fig.add_trace(go.Scatter3d(
                        x=[p168_wc[0], p168_wc[0]+Vf[0]*0.054], y=[p168_wc[1], p168_wc[1]+Vf[1]*0.054], z=[p168_wc[2], p168_wc[2]+Vf[2]*0.054], 
                        mode='lines', line=dict(color='red', width=10), 
                        name='HƯỚNG NHÌN TỔNG HỢP [3]', meta='3'
                    ))

                    
                    log_data.append(f"\n[FINAL GAZE]")
                    log_data.append(f"  V_final: {Vf}")
                    log_data.append(f"  Yaw:   {st_yaw}° | Pitch: {st_pitch}°\n")

            # Write log to file
            log_out = OUTPUT_VIS_3D / f"3d_log_{path.stem}.txt"
            with open(log_out, "w", encoding="utf-8") as f:
                f.write("\n".join(log_data))

            # 4. Landmark mặt (Nhóm 1)
            # Dùng FACEMESH_TESSELATION để vẽ lưới tam giác liên kết toàn bộ mặt
            import mediapipe as mp
            x_lines, y_lines, z_lines = [], [], []
            for start, end in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                if start in wc_d and end in wc_d:
                    x_lines.extend([wc_d[start][0], wc_d[end][0], None])
                    y_lines.extend([wc_d[start][1], wc_d[end][1], None])
                    z_lines.extend([wc_d[start][2], wc_d[end][2], None])
            
            if x_lines:
                fig.add_trace(go.Scatter3d(
                    x=x_lines, y=y_lines, z=z_lines, 
                    mode='lines', line=dict(color='gray', width=1), 
                    opacity=0.3, name='Lưới Khuôn Mặt [1]', meta='1', hoverinfo='skip'
                ))
            
            # Mesh xám mờ + text ID rời
            upper_ids = [idx for idx in range(468) if idx in wc_d and (idx < 10 or (33 <= idx < 160) or (263 <= idx < 390) or idx in KEY_IDS)]
            pts = np.array([wc_d[idx] for idx in upper_ids])
            if pts.size > 0:
                mode_3d = 'markers+text' if GLOBAL_CONFIG["show_ids"] else 'markers'
                txt_3d  = [str(idx) for idx in upper_ids] if GLOBAL_CONFIG["show_ids"] else None
                fig.add_trace(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2], 
                    mode=mode_3d, marker=dict(size=2, color='gray', opacity=0.4), 
                    text=txt_3d, textposition="top center", textfont=dict(size=7, color='black'),
                    name='Face Mesh Markers [1]', meta='1'
                ))

            # Layout config
            camera = dict(eye=dict(x=0, y=0, z=-1.5), up=dict(x=0, y=1, z=0))
            fig.update_layout(
                scene=dict(
                    camera=camera,
                    xaxis=dict(range=[pts[:,0].min()-0.05, pts[:,0].max()+0.05] if pts.size > 0 else None, autorange=False, title='X (Ngang)'),
                    yaxis=dict(range=[pts[:,1].min()-0.05, pts[:,1].max()+0.05] if pts.size > 0 else None, autorange=False, title='Y (Dọc)'),
                    zaxis=dict(range=[pts[:,2].min()-0.05, pts[:,2].max()+0.05] if pts.size > 0 else None, autorange=False, title='Z (Sâu)'),
                    aspectmode='cube'
                ),
                title=f"3D Gaze Analysis: {path.name} | Gaze: ({st_yaw}°, {st_pitch}°)",
                margin=dict(l=0, r=0, b=0, t=40)
            )

            # Thêm HDSD lên góc trên biểu đồ
            fig.add_annotation(
                text="<b>Bấm phím 1</b>: Lưới Mặt | <b>2</b>: Mắt & Đồng tử | <b>3</b>: Tính toán Gaze",
                x=0.01, y=0.98, xref="paper", yref="paper", showarrow=False,
                font=dict(size=14, color="white"), bgcolor="rgba(0,0,0,0.6)", borderpad=6
            )

            out_html = OUTPUT_VIS_3D / f"3d_{path.stem}.html"
            fig.write_html(str(out_html))
            
            # Tiêm thêm mã JS ngầm vào cuối tệp HTML để xử lý sự kiện gõ phím '1','2','3'
            custom_js = """
            <script>
            document.addEventListener('keydown', function(event) {
                var key = event.key;
                if (!['1', '2', '3'].includes(key)) return;
                
                var plots = document.getElementsByClassName('plotly-graph-div');
                for (var p = 0; p < plots.length; p++) {
                    var gd = plots[p];
                    if (!gd || !gd.data) continue;
                    
                    var update = {visible: []};
                    var indices = [];
                    for (var i = 0; i < gd.data.length; i++) {
                        if (gd.data[i].meta === key) {
                            indices.push(i);
                            var currentVis = gd.data[i].visible;
                            // Nút chuyển trạng thái huyền ảo: nếu đang bật thì ẩn (legendonly hoặc false), ngược lại bật true
                            if (currentVis === undefined || currentVis === true) {
                                update.visible.push('legendonly'); 
                            } else {
                                update.visible.push(true);
                            }
                        }
                    }
                    if (indices.length > 0) {
                        Plotly.restyle(gd, update, indices);
                    }
                }
            });
            </script>
            """
            with open(out_html, 'a', encoding='utf-8') as html_file:
                html_file.write(custom_js)

            print(f"    [✓] Đã tạo HTML: {out_html.name}")

        except Exception as e:
            print(f"    [ERR] {path.name}: {e}")

    fm.close()
    print(f"\n[DONE] Hoàn tất 3D Visualization. Kết quả tại {OUTPUT_VIS_3D}\n")

def run_video():
    import numpy as np
    from ultralytics import YOLO
    print("\n[MODE] Video Processing")
    
    # Tìm file trong input/video/
    v_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    v_files = []
    for ext in v_exts:
        v_files.extend(list(INPUT_VIDEO.glob(ext)))
    v_files.sort()
    
    to_process = []
    if v_files:
        print(f"\n  [*] Danh sách video trong {INPUT_VIDEO.name}:")
        for i, f in enumerate(v_files):
            print(f"      {i+1}. {f.name}")
        print(f"      A. Chạy tất cả ({len(v_files)} video)")
        print(f"      F. Chọn file khác từ máy tính...")
        print(f"      0. Quay lại")
        
        choice = input("\n  [?] Lựa chọn (1-{0}, A, F, 0): ".format(len(v_files))).strip().upper()
        
        if choice == '0': return
        elif choice == 'A': to_process = v_files
        elif choice == 'F': pass # Sẽ mở tkinter bên dưới
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(v_files):
                    to_process = [v_files[idx]]
                else: 
                    print("  [!] Số thứ tự không hợp lệ."); return
            except ValueError:
                print("  [!] Lựa chọn không hợp lệ."); return
    
    if not to_process:
        # Nếu chưa có video nào được chọn (folder trống HOẶC người dùng chọn F)
        print(f"  [*] Mở hộp thoại chọn file...")
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        vpath = filedialog.askopenfilename(title="Chọn file video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        root.destroy()
        if not vpath: print("  [!] Không có file nào được chọn."); return
        to_process = [Path(vpath)]


    dev = get_device()
    use_onnx = GLOBAL_CONFIG.get("use_onnx", True)
    
    # 1. Khởi tạo Engine
    onnx_engine = None
    yolo_model = None
    fm_mp = None
    
    if use_onnx:
        try:
            onnx_engine = ONNXGazeEngine()
            print("  [✓] Đã khởi tạo ONNX GPU Engine.")
        except Exception as e:
            print(f"  [!] Lỗi khởi tạo ONNX: {e}. Fallback sang MediaPipe.")
            use_onnx = False

    if not use_onnx:
        yolo_model = YOLO(str(YOLO_FACE_MODEL)).to(dev)
        fm_mp = make_face_mesh(static=False)
        print(f"  [*] Khởi tạo MediaPipe + YOLO trên: {dev.upper()}")

    print("\n  [?] Chọn chế độ xử lý:")
    print("      1. Fast Mode (720p, High FPS, Có Preview)")
    print("      2. Ultra-Safe (Low-res, No-Preview, Ổn định cao)")
    safe_choice = input("      Lựa chọn (1 hoặc 2): ").strip()
    
    use_preview = (safe_choice != "2")
    yolo_imgsz = 320 if safe_choice == "2" else 640
    # ... logic tiếp theo sử dụng engine đã chọn
    MAX_PROC_W = 640 if safe_choice == "2" else 1080

    for idx, vpath in enumerate(to_process):
        print(f"\n  [{idx+1}/{len(to_process)}] Đang xử lý: {vpath.name}")
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened(): print(f"    [ERR] Không mở được video {vpath.name}"); continue
        
        vw_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        vtotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_w, target_h = vw_orig, vh_orig
        if vw_orig > MAX_PROC_W:
            scale = MAX_PROC_W / vw_orig
            target_w, target_h = int(vw_orig * scale), int(vh_orig * scale)
            print(f"    [INFO] Resize: {vw_orig}x{vh_orig} -> {target_w}x{target_h}")

        out_path = OUTPUT_VIDEO / f"proc_{vpath.name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, vfps, (target_w, target_h))
        
        print(f"    [*] Bắt đầu: {vtotal} frames. Nhấn 'Q' để dừng video này.")

        show_mesh, show_ids = True, GLOBAL_CONFIG["show_ids"]
        curr_f = 0
        tracker = FaceTracker()
        last_tracked = []
        fm_shared = None
        v_start_t = time.time()
        v_prev_t = v_start_t
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None: break
                curr_f += 1
                if vw_orig != target_w: frame = cv2.resize(frame, (target_w, target_h))

                multi = GLOBAL_CONFIG["multi_face"]
                if multi:
                    interval = GLOBAL_CONFIG.get("detect_interval", 3)
                    if curr_f % interval == 0 or not last_tracked:
                        if use_onnx:
                            boxes = onnx_engine.detect(frame, conf_threshold=GLOBAL_CONFIG["yolo_conf"])
                        else:
                            res = yolo_model(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=yolo_imgsz, verbose=False)[0]
                            boxes = res.boxes
                        last_tracked = tracker.update(boxes)
                    rows, _ = process_frame(frame, last_tracked, tracker=tracker, 
                                            face_mesh=onnx_engine if use_onnx else None, 
                                            show_mesh=show_mesh, show_ids=show_ids)
                else:
                    # Fast Mode (Single Face)
                    if not use_onnx and not fm_mp: fm_mp = make_face_mesh(static=False)
                    if use_onnx:
                        boxes = onnx_engine.detect(frame, conf_threshold=GLOBAL_CONFIG["yolo_conf"])
                    else:
                        res = yolo_model(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=yolo_imgsz, verbose=False)[0]
                        boxes = res.boxes
                        
                    if boxes:
                        if use_onnx:
                            best_box = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                        else:
                            best_box = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                            
                        rows, _ = process_frame(frame, [(best_box, 0)], 
                                                face_mesh=onnx_engine if use_onnx else fm_mp, 
                                                show_mesh=show_mesh, show_ids=show_ids)
                    else: rows = []

                # Tính FPS xử lý
                now = time.time()
                proc_fps = 1.0 / (now - v_prev_t + 1e-9)
                v_prev_t = now

                writer.write(frame)
                
                if use_preview:
                    cv2.putText(frame, f"Proc FPS: {proc_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Processing Video (Q to Skip)", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord('q'), 27): 
                        print(f"\n    [!] Dừng video: {vpath.name} (phím bấm)")
                        break
                    # window property check: đợi sau 10 frames để window kịp khởi tạo
                    if curr_f > 10:
                        try:
                            if cv2.getWindowProperty("Processing Video (Q to Skip)", cv2.WND_PROP_VISIBLE) < 1: 
                                print(f"\n    [!] Dừng video: {vpath.name} (đóng cửa sổ)")
                                break
                        except: break
                else:
                    if curr_f % 100 == 0: time.sleep(0.001)

                if curr_f % 20 == 0 or curr_f == vtotal:
                    p = (curr_f / vtotal) * 100 if vtotal > 0 else 0
                    avg_fps = curr_f / (time.time() - v_start_t + 1e-9)
                    print(f"      -> {vpath.name}: {curr_f}/{vtotal} ({p:.1f}%) | FPS: {proc_fps:.1f} (Avg: {avg_fps:.1f})", end='\r')
                
                # Throttle nhỏ để OS không bị treo hoàn toàn
                time.sleep(0.001)
                
        except Exception as e:
            print(f"\n    [ERR] Lỗi khi xử lý {vpath.name}: {e}")
            import traceback
            traceback.print_exc()
            
        cap.release(); writer.release()
        if fm_shared: fm_shared.close()
        print(f"\n    [✓] Xong: {out_path.name}")
        
    cv2.destroyAllWindows()
    print(f"\n[DONE] Hoàn tất xử lý video.\n")

def run_toggle_ids():
    GLOBAL_CONFIG["show_ids"] = not GLOBAL_CONFIG["show_ids"]
    print(f"  [CONFIG] Landmark IDs: {'BẬT' if GLOBAL_CONFIG['show_ids'] else 'TẮT'}")

def run_toggle_gaze_mode():
    GLOBAL_CONFIG["use_eye_gaze"] = not GLOBAL_CONFIG["use_eye_gaze"]
    print(f"  [CONFIG] Chế độ: {'Mặt+Mắt' if GLOBAL_CONFIG['use_eye_gaze'] else 'Chỉ Mặt'}")

def run_toggle_device():
    import torch
    GLOBAL_CONFIG["force_device"] = "cpu" if GLOBAL_CONFIG["force_device"] == "cuda" else "cuda"
    print(f"  [CONFIG] Thiết bị: {GLOBAL_CONFIG['force_device'].upper()}")

def run_toggle_multi_face():
    GLOBAL_CONFIG["multi_face"] = not GLOBAL_CONFIG["multi_face"]
    print(f"  [CONFIG] Chế độ: {'Đa khuôn mặt (Robust)' if GLOBAL_CONFIG['multi_face'] else 'Đơn khuôn mặt (Fast)'}")

def run_settings_menu():
    while True:
        print(f"\n{'='*40}\n  CÀI ĐẶT THAM SỐ\n{'='*40}")
        print(f"  1. Smooth Alpha [{GLOBAL_CONFIG['smooth_alpha']}]")
        print(f"  2. YOLO Conf [{GLOBAL_CONFIG['yolo_conf']}]")
        print(f"  3. ROI Padding [{GLOBAL_CONFIG['roi_padding']}]")
        print(f"  4. Detect Interval [{GLOBAL_CONFIG['detect_interval']}] (mỗi N frame)")
        print(f"  5. High FPS Mode [{'BẬT' if GLOBAL_CONFIG['high_fps_mode'] else 'TẮT'}]")
        print(f"  6. Engine: [{'ONNX (GPU)' if GLOBAL_CONFIG['use_onnx'] else 'MediaPipe (CPU)'}]")
        print(f"  0. Quay lại")
        c = input("\nChọn: ").strip()
        if c == '0': break
        try:
            if c == '5':
                GLOBAL_CONFIG['high_fps_mode'] = not GLOBAL_CONFIG['high_fps_mode']
                print("  [OK]")
                continue
            if c == '6':
                GLOBAL_CONFIG['use_onnx'] = not GLOBAL_CONFIG['use_onnx']
                print("  [OK]")
                continue

            v_str = input("Nhập giá trị mới: ")
            v = float(v_str)
            if c == '1': GLOBAL_CONFIG['smooth_alpha'] = v
            elif c == '2': GLOBAL_CONFIG['yolo_conf'] = v
            elif c == '3': GLOBAL_CONFIG['roi_padding'] = v
            elif c == '4': GLOBAL_CONFIG['detect_interval'] = int(v)
            print("  [OK]")
        except: print("  [!]")

def run_status_report():
    import platform
    import onnxruntime as ort
    import torch
    
    print(f"\n{'='*55}")
    print(f"{'HỆ THỐNG GAZE ESTIMATION - THÔNG TIN CHI TIẾT':^55}")
    print(f"{'='*55}")
    
    # 1. PHẦN MỀM & PHẦN CỨNG
    print("\n[1] MÔI TRƯỜNG & PHẦN MỀM")
    print(f"  - OS:           {platform.system()} {platform.release()}")
    print(f"  - Python:       {platform.python_version()}")
    print(f"  - OpenCV:       {cv2.__version__}")
    print(f"  - ONNX Runtime: {ort.__version__}")
    print(f"  - PyTorch:      {torch.__version__}")
    print(f"  - CUDA:         {'CÓ (Sẵn sàng)' if torch.cuda.is_available() else 'KHÔNG (CPU Mode)'}")
    if torch.cuda.is_available():
        print(f"  - GPU Device:   {torch.cuda.get_device_name(0)}")
    
    # 2. CẤU HÌNH INFERENCE
    print("\n[2] CẤU HÌNH INFERENCE (GLOBAL_CONFIG)")
    print(f"  - Engine:       {'ONNX Runtime (GPU)' if GLOBAL_CONFIG['use_onnx'] else 'MediaPipe (CPU/Fallback)'}")
    print(f"  - Strategy:     {'Robust (Multi-Face)' if GLOBAL_CONFIG['multi_face'] else 'Fast (Single-Face)'}")
    print(f"  - Device Force: {GLOBAL_CONFIG['force_device'].upper()}")
    print(f"  - YOLO Conf:    {GLOBAL_CONFIG['yolo_conf']}")
    print(f"  - Detect Int:   {GLOBAL_CONFIG['detect_interval']} frames (YOLO Skip)")
    print(f"  - High FPS:     {'BẬT (256px, No-PP)' if GLOBAL_CONFIG['high_fps_mode'] else 'TẮT (384px, CLAHE+Sharp)'}")
    
    # 3. THÔNG SỐ HÌNH HỌC (GAZE GEOMETRY)
    print("\n[3] THÔNG SỐ HÌNH HỌC & LÀM MƯỢT")
    print(f"  - Smooth Alpha: {GLOBAL_CONFIG['smooth_alpha']} (Smoothing strength)")
    print(f"  - ROI Padding:  {GLOBAL_CONFIG['roi_padding']} (Crop buffer)")
    print(f"  - Gaze Mode:    {'Mặt + Mống mắt' if GLOBAL_CONFIG['use_eye_gaze'] else 'Chỉ hướng mặt'}")
    print(f"  - Show IDs:     {GLOBAL_CONFIG['show_ids']}")
    
    # 4. TRẠNG THÁI MÔ HÌNH (MODEL FILES)
    print("\n[4] KIỂM TRA TỆP MÔ HÌNH")
    models = {
        "YOLO PT": BASE_DIR / "yolov8_head" / "yolov8n-face.pt",
        "YOLO ONNX": BASE_DIR / "c_ver" / "yolov8n-face.onnx",
        "FaceMesh ONNX": BASE_DIR / "c_ver" / "facemesh_refined.onnx"
    }
    for name, path in models.items():
        status = "OK" if path.exists() else "THIẾU"
        size = f"({path.stat().st_size/1024/1024:.1f} MB)" if path.exists() else ""
        print(f"  - {name:<14}: {status} {size}")

    # 5. ĐƯỜNG DẪN DỮ LIỆU
    print("\n[5] ĐƯỜNG DẪN (DIRECTORIES)")
    print(f"  - Input Dir:    {INPUT_DIR.relative_to(BASE_DIR)}")
    print(f"  - Video Output: {OUTPUT_VIDEO.relative_to(BASE_DIR)}")
    print(f"  - Data Log:     {DATA_DIR.relative_to(BASE_DIR)}")

    print(f"\n{'='*55}")
    input("Nhấn ENTER để quay lại menu chính...")

# ------------------------------------------------------------------ #
#  MENU                                                                #
# ------------------------------------------------------------------ #
def get_menu():
    st_ids = "BẬT" if GLOBAL_CONFIG["show_ids"] else "TẮT"
    st_gaze = "Mặt+Mắt" if GLOBAL_CONFIG["use_eye_gaze"] else "Chỉ Mặt"
    st_mode = "Robust" if GLOBAL_CONFIG["multi_face"] else "Fast"
    dev = GLOBAL_CONFIG["force_device"].upper()
    hw = GLOBAL_CONFIG["hw_detected"]
    
    return f"""
╔══════════════════════════════════════════╗
║    GEOMETRIC GAZE ESTIMATION             ║
╠══════════════════════════════════════════╣
║ Hardware: {hw:<31}║
╠══════════════════════════════════════════╣
║  1  Batch  (ảnh trong input/)            ║
║  2  Webcam (real-time camera)            ║
║  3  Vis 2D (debug hình học trên ảnh)     ║
║  4  Vis 3D (mô hình 3D tương tác)        ║
║  5  Video  (xử lý file video)            ║
║  6  Cấu hình Landmark IDs [{st_ids}]     ║
║  7  Chế độ Gaze [{st_gaze}]              ║
║  8  Thiết bị [{dev}]                     ║
║  9  Chế độ [{st_mode}] (Robust/Fast)     ║
║  S  Cài đặt tham số nâng cao             ║
║  D  Thông số hệ thống chi tiết           ║
║  0  Thoát                                ║
╚══════════════════════════════════════════╝
Chọn: """
HANDLERS = {
    '1': run_batch, 
    '2': run_webcam, 
    '3': run_vis2d, 
    '4': run_vis3d,
    '5': run_video,
    '6': run_toggle_ids,
    '7': run_toggle_gaze_mode,
    '8': run_toggle_device,
    '9': run_toggle_multi_face, # Phím 9 chuyển mode
    'S': run_settings_menu,     # Phím S cho cài đặt
    'D': run_status_report      # Phím D cho cấu hình chi tiết
}

if __name__ == "__main__":
    while True:
        try:
            choice = input(get_menu()).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nThoát."); break
        if choice == '0':
            print("Thoát."); break
        elif choice in HANDLERS:
            HANDLERS[choice]()
        else:
            print("  Lựa chọn không hợp lệ, thử lại.\n")
