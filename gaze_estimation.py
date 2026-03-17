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

import cv2
import sys
import csv
import time
from pathlib import Path

# Các thư viện nặng sẽ được lazy-load trong hàm (torch, numpy, plotly, mediapipe, yolo)

# ------------------------------------------------------------------ #
#  CẤU HÌNH                                                           #
# ------------------------------------------------------------------ #
BASE_DIR        = Path(__file__).resolve().parent   # d:\MMPOSE
YOLO_FACE_MODEL = BASE_DIR / "yolov8_head" / "yolov8n-face.pt"

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

CAMERA_ID    = 0
GLOBAL_CONFIG = {
    "show_ids": False,     # Mặc định TẮT Landmark IDs
    "smooth_alpha": 0.8,   # 0.1: mượt/trễ, 0.9: nhạy/rung
    "use_eye_gaze": False, # Mặc định Chỉ Mặt
    "force_device": "cuda",# Mặc định chạy CUDA
    "yolo_conf": 0.35,     # Ngưỡng tin cậy YOLO
    "S_H": 28.0,           # Độ nhạy ngang
    "S_V": 12.35           # Độ nhạy dọc
}

# Màu sắc mặc định cho toàn bộ (Green - BGR)
FACE_COLOR = (0, 255, 0)


class FaceTracker:
    def __init__(self, iou_threshold=0.3, max_lost=15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}  # {id: {"box": [x1,y1,x2,y2], "lost": 0, "smooth_gaze": None}}
        self.next_id = 0

    def update(self, yolo_res):
        if yolo_res is None or not len(yolo_res):
            for tid in self.tracks: self.tracks[tid]["lost"] += 1
            self._cleanup()
            return []

        new_boxes = []
        for box in yolo_res:
            new_boxes.append(box.xyxy[0].tolist())

        matched_indices = {} # new_box_idx -> track_id
        
        # 1. Match with existing tracks
        for tid, track in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for b_idx, box in enumerate(new_boxes):
                if b_idx in matched_indices: continue
                iou = self._iou(track["box"], box)
                if iou > best_iou:
                    best_iou, best_idx = iou, b_idx
            
            if best_iou > self.iou_threshold:
                matched_indices[best_idx] = tid
                self.tracks[tid]["box"] = new_boxes[best_idx]
                self.tracks[tid]["lost"] = 0
            else:
                self.tracks[tid]["lost"] += 1

        # 2. Add new tracks
        for b_idx, box in enumerate(new_boxes):
            if b_idx not in matched_indices:
                self.tracks[self.next_id] = {"box": box, "lost": 0, "smooth_gaze": None}
                matched_indices[b_idx] = self.next_id
                self.next_id += 1

        self._cleanup()

        # 3. Return results: list of (original_box_obj, track_id)
        results = []
        for b_idx, box in enumerate(new_boxes):
            results.append((yolo_res[b_idx], matched_indices[b_idx]))
        return results

    def get_smooth_gaze(self, tid):
        return self.tracks.get(tid, {}).get("smooth_gaze")

    def set_smooth_gaze(self, tid, gaze_vec):
        if tid in self.tracks:
            self.tracks[tid]["smooth_gaze"] = gaze_vec

    def _cleanup(self):
        to_del = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in to_del: del self.tracks[tid]

    def _iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / float(area1 + area2 - inter + 1e-6)

def get_device():
    import torch
    if GLOBAL_CONFIG["force_device"] == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_face_mesh(static=False):
    import mediapipe as mp
    _mp_fm = mp.solutions.face_mesh
    return _mp_fm.FaceMesh(
        static_image_mode      = static,
        max_num_faces          = 1,
        refine_landmarks       = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

def get_iris_connections():
    import mediapipe as mp
    return mp.solutions.face_mesh.FACEMESH_IRISES

KEY_IDS = [163, 157, 161, 154, 468, 390, 384, 388, 381, 473, 168]

# ------------------------------------------------------------------ #
#  HÌNH HỌC                                                           #
# ------------------------------------------------------------------ #
def shortest_common_perpendicular(P1, P2, P3, P4):
    import numpy as np
    """
    Tìm giao điểm ảo (O): Là trung điểm của đoạn vuông góc chung ngắn nhất giữa L1 (P1-P2) và L2 (P3-P4).
    """
    u1 = P2 - P1; n1 = np.linalg.norm(u1)
    if n1 < 1e-6: return None, None, None
    u1 /= n1
    u2 = P4 - P3; n2 = np.linalg.norm(u2)
    if n2 < 1e-6: return None, None, None
    u2 /= n2
    
    # dp = P1 - P3
    dp = P1 - P3
    dot12 = np.dot(u1, u2)
    dot1p = np.dot(u1, dp)
    dot2p = np.dot(u2, dp)
    
    det = 1.0 - dot12**2
    if det < 1e-7: # Song song
        return (P1 + P3) / 2, u1, u2

    s = (dot12 * dot2p - dot1p) / det
    t = (dot2p - dot12 * dot1p) / det
    
    C1 = P1 + s * u1
    C2 = P3 + t * u2
    O = (C1 + C2) / 2
    return O, u1, u2

def get_local_axes(u1, u2):
    import numpy as np
    """
    Thiết lập Hệ tọa độ Địa phương (Ox, Oy) bằng hai đường phân giác của L1, L2.
    Phân loại dựa trên thành phần chiếm ưu thế để đảm bảo Ox là Ngang và Oy là Dọc.
    """
    b1 = u1 + u2; b2 = u1 - u2 # Hai đường phân giác (chưa chuẩn hóa)
    n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
    b1 = b1/n1 if n1 > 1e-6 else u1
    b2 = b2/n2 if n2 > 1e-6 else u2
    
    # Phân loại trục dựa trên thành phần chiếm ưu thế (X hay Y)
    # (Hệ MediaPipe World: X+ Right, Y+ Up)
    if abs(b1[0]) > abs(b1[1]): # b1 thiên về phương ngang hơn
        ox, oy = b1, b2
    else: # b1 thiên về phương dọc hơn
        ox, oy = b2, b1
        
    # CHUẨN HÓA HƯỚNG: Ox sang TRÁI hốc mắt (Subject's Right), Oy lên TRÊN (Subject's Up)
    # LƯU Ý: Landmark 102 (Right) có X nhỏ hơn 331 (Left), nên Subject's Right là hướng X âm.
    if ox[0] > 0: ox = -ox
    if oy[1] < 0: oy = -oy
    
    return ox, oy

def process_eye(P1, P2, P3, P4, pupil):
    import numpy as np
    O, u1, u2 = shortest_common_perpendicular(P1, P2, P3, P4)
    if O is None: return None, None
    Ox, Oy = get_local_axes(u1, u2)
    V = pupil - O
    
    # Chiếu pupil xuống mặt phẳng (Ox, Oy) qua O và tính tọa độ x0, y0
    x0 = np.dot(V, Ox)
    y0 = np.dot(V, Oy)
    
    alpha = np.degrees(np.arctan2(y0, x0))
    d = np.linalg.norm([x0, y0])
    return alpha, d

def _pt(world, idx):
    import numpy as np
    return np.array(world[idx]) if idx in world else None

def calculate_gaze(world):
    import numpy as np
    use_eye = GLOBAL_CONFIG.get("use_eye_gaze", True)
    
    if not use_eye:
        # TỐI ƯU: Chỉ lấy 3 điểm để tính hướng mặt
        p_face = {k: _pt(world, v) for k, v in dict(g=168, a=331, b=102).items()}
        if any(v is None for v in p_face.values()): return None, None, None, None
        
        v_left  = p_face['a'] - p_face['g']
        v_right = p_face['b'] - p_face['g']
        nf = np.cross(v_left, v_right)
        nn = np.linalg.norm(nf)
        V_face = nf/nn if nn > 1e-6 else np.array([0,0,-1.])
        if V_face[2] > 0: V_face = -V_face
        return None, None, p_face['g'], V_face

    # Chế độ đầy đủ: Mặt + Mắt
    ids = dict(l1=163,l2=157,l3=161,l4=154,lp=468,
               r1=390,r2=384,r3=388,r4=381,rp=473,
               g=168, a=331, b=102)
    p = {k: _pt(world, v) for k, v in ids.items()}
    if any(v is None for v in p.values()): return None, None, None, None

    aL, dL = process_eye(p['l1'], p['l2'], p['l3'], p['l4'], p['lp'])
    aR, dR = process_eye(p['r1'], p['r2'], p['r3'], p['r4'], p['rp'])
    if aL is None or aR is None: return None, None, None, None

    alpha_avg = (aL + aR) / 2.0
    d_avg     = (dL + dR) / 2.0

    # Trục khuôn mặt: V_face, Rf (Right), Uf (Up)
    # 1. V_face: Pháp tuyến mặt phẳng 168(g), 102(b), 331(a)
    # Z MediaPipe World: + là Back (vào trong màn hình), - là Front (về phía camera).
    # Ta cần V_face hướng về Camera (Z âm).
    v_left  = p['a'] - p['g'] # 331 (Trái mẫu) - 168
    v_right = p['b'] - p['g'] # 102 (Phải mẫu) - 168
    
    nf = np.cross(v_left, v_right)
    nn = np.linalg.norm(nf)
    V_face = nf/nn if nn > 1e-6 else np.array([0,0,-1.])
    if V_face[2] > 0: V_face = -V_face # Ép hướng Front

    # 2. Rf (Right): Hướng từ Trái sang Phải hốc mắt (X dương)
    Rf = p['b'] - p['a'] # Landmarks 102 - 331
    Rf = Rf - np.dot(Rf, V_face) * V_face
    nr = np.linalg.norm(Rf)
    Rf = Rf/nr if nr > 1e-6 else np.array([1.,0,0])
    
    # 3. Uf (Up): V_face là Front (Z-), Rf là Right (X+). 
    # Theo quy tắc bàn tay phải (X, Y, Z), Y = Z x X.
    Uf = np.cross(V_face, Rf)
    if Uf[1] < 0: Uf = -Uf # Đảm bảo hướng Up (Y dương)

    # Phép cộng Vector (Bẻ hướng)
    # GAZE_SENSITIVITY: 
    S_H = GLOBAL_CONFIG.get("S_H", 28.0)
    S_V = GLOBAL_CONFIG.get("S_V", 12.35)
    
    ar = np.radians(alpha_avg)
    # Bẻ hướng theo từng trục độc lập
    V_final_unnorm = V_face + d_avg * (np.cos(ar) * S_H * Rf + np.sin(ar) * S_V * Uf)
    nv = np.linalg.norm(V_final_unnorm)
    V_final = V_final_unnorm / nv if nv > 1e-6 else V_face
    
    return alpha_avg, d_avg, p['g'], V_final

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
    # Phóng đại Vx, Vy để vẽ hướng trên ảnh. Kéo dãn gấp 2 lần (2.0)
    L = min(w_c, h_c) * 2.0
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
    """Vẽ L1, L2, O, Ox, Oy cho 2 mắt (chế độ Vis 2D)."""
    for ids in [(163,157,161,154,468), (390,384,388,381,473)]:
        p1,p2,p3,p4,_ = [_pt(wc, i) for i in ids]
        if any(x is None for x in [p1,p2,p3,p4]): continue
        O, u1, u2 = shortest_common_perpendicular(p1,p2,p3,p4)
        if O is None: continue
        Ox, Oy = get_local_axes(u1, u2)
        if ids[0] in px and ids[2] in px:
            cv2.line(frame, px[ids[0]], px[ids[1]], (255,200,0), 2)
            cv2.line(frame, px[ids[2]], px[ids[3]], (255,200,0), 2)
        # Ox, Oy arrows (scaled visually)
        def _wc2px(pt3d, ref_idx, wc, px):
            if ref_idx not in wc or ref_idx not in px: return None
            ref_w = np.array(wc[ref_idx])
            ref_p = px[ref_idx]
            dp = pt3d - ref_w
            sc = 800
            return (int(ref_p[0] + dp[0]*sc), int(ref_p[1] + dp[1]*sc))
        origin_px = px.get(ids[0])
        if origin_px:
            scale = 0.04
            ex = _wc2px(O + Ox*scale, ids[0], wc, px)
            ey = _wc2px(O + Oy*scale, ids[0], wc, px)
            op = _wc2px(O, ids[0], wc, px)
            if op:
                cv2.drawMarker(frame, op, (0,0,255), cv2.MARKER_CROSS, 12, 2)
                if ex: cv2.arrowedLine(frame, op, ex, (0,200,0), 2, tipLength=0.35)
                if ey: cv2.arrowedLine(frame, op, ey, (200,0,200), 2, tipLength=0.35)

def draw_key_ids(frame, px):
    for i in KEY_IDS:
        if i in px:
            cv2.putText(frame, str(i), px[i], cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 1)

def draw_hud(frame, fps, alpha, dist, show_mesh, show_ids):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (10,10), (330,135), (0,0,0), -1)
    cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Alpha: {alpha:.1f} deg" if alpha else "Alpha: --", (20,62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Dist:  {dist:.4f}" if dist else "Dist:  --", (20,87), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Mesh:{'ON' if show_mesh else 'OFF'} IDs:{'ON' if show_ids else 'OFF'}", (20,112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, "Q/ESC Quit | S Save | M Mesh | I IDs", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

# ------------------------------------------------------------------ #
#  SHARED FRAME PROCESSOR (Batch + Webcam dùng chung)                  #
# ------------------------------------------------------------------ #
def process_frame(frame, tracked_faces, face_mesh, tracker=None, show_mesh=True, show_ids=False, vis2d=False):
    """
    Xử lý 1 frame: landmark → tính gaze → vẽ dựa trên tracked_faces [(box, id), ...].
    Hỗ trợ làm mượt riêng biệt cho từng khuôn mặt nếu có tracker.
    Trả về: (rows, None)
    """
    import numpy as np
    oh, ow = frame.shape[:2]
    rows = []
    current_v = None

    if not tracked_faces:
        return rows, None  # không có mặt

    for box, tid in tracked_faces:
        color = FACE_COLOR
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(ow,x2), min(oh,y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        hc, wc = crop.shape[:2]

        # VẼ KHUNG TRƯỚC: Đảm bảo luôn thấy box nếu YOLO detect được
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 1)

        mpr = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not mpr.multi_face_landmarks: continue

        wlms = getattr(mpr, 'multi_face_world_landmarks', None)
        px, wc_d = build_coords(
            mpr.multi_face_landmarks[0],
            wlms[0] if wlms else None,
            x1, y1, wc, hc
        )

        # --- Vẽ tùy chế độ ---
        # Bỏ vẽ Iris theo yêu cầu
        if vis2d:
            draw_iris(frame, px)
            draw_eye_geometry(frame, px, wc_d)
        if show_ids or vis2d:
            draw_key_ids(frame, px)

        # --- Tính & vẽ gaze ---
        alpha_val, d_val, _, Vf = calculate_gaze(wc_d)
        if Vf is not None:
            # EMA Smoothing riêng cho từng khuôn mặt (ID)
            if tracker is not None:
                prev_v = tracker.get_smooth_gaze(tid)
                if prev_v is not None:
                    alpha = GLOBAL_CONFIG.get("smooth_alpha", 0.3)
                    Vf = alpha * Vf + (1 - alpha) * prev_v
                    nv = np.linalg.norm(Vf)
                    if nv > 1e-6: Vf /= nv
                tracker.set_smooth_gaze(tid, Vf)
            
            draw_arrow(frame, px, Vf, wc, hc, color=color)
            
            # Chỉ hiển thị text Alpha/D nếu có giá trị (chế độ Mặt+Mắt)
            txt = f"A:{alpha_val:.1f} D:{d_val:.4f}" if alpha_val is not None else "FACE MODE"
            cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            rows.append({
                "alpha_degrees": alpha_val if alpha_val is not None else 0, 
                "distance_to_O": d_val if d_val is not None else 0,
                "final_gaze_vector": f"({Vf[0]:.4f}, {Vf[1]:.4f}, {Vf[2]:.4f})"
            })

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
        # Sử dụng GLOBAL_CONFIG.show_ids, không smoothing cho batch (vì ảnh rời rạc)
        res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
        tracked = [(box, i) for i, box in enumerate(res.boxes)] if res.boxes else []
        rows, _ = process_frame(frame, tracked, fm, show_mesh=True, show_ids=GLOBAL_CONFIG["show_ids"])
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
#  CHẾ ĐỘ 2 - WEBCAM                                                  #
# ------------------------------------------------------------------ #
def run_webcam():
    from ultralytics import YOLO
    print("\n[MODE] Webcam Real-time")
    dev = get_device()
    print(f"  [*] Khởi tạo Face Detection trên: {dev.upper()}")
    yolo = YOLO(str(YOLO_FACE_MODEL)).to(dev)
    print("  [*] Khởi tạo Landmark Extractor trên: CPU (MediaPipe)")
    fm   = make_face_mesh(static=False)
    # Thử mở camera với các backend khác nhau nếu lỗi
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID) # Fallback
        if not cap.isOpened():
            print(f"  [ERR] Không mở được camera ID {CAMERA_ID}."); return
            
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    
    # Một số camera cần thời gian khởi động
    print("  [*] Đang khởi động camera...")
    for _ in range(15):
        ret, _ = cap.read()
        if ret: break
        time.sleep(0.1)
    
    if not ret: print("  [ERR] Camera không truyền được dữ liệu."); cap.release(); return
    print("  [*] Webcam đã sẵn sàng.")

    show_mesh, show_ids = True, GLOBAL_CONFIG["show_ids"]
    prev_t, save_n = time.time(), 0
    last_a, last_d = None, None

    print("  [*] Bắt đầu hiển thị. Nhấn 'Q' để thoát.")
    last_v = None
    tracker = FaceTracker()
    last_tracked_faces = []
    frame_idx = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None: 
                print("  [ERR] Không nhận được frame từ camera.")
                break
            
            frame_idx += 1
            # YOLO mỗi 5 frames
            if frame_idx % 5 == 0 or not last_tracked_faces:
                res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=640, verbose=False)[0]
                last_tracked_faces = tracker.update(res.boxes)

            rows, _ = process_frame(frame, last_tracked_faces, fm, tracker=tracker, show_mesh=show_mesh, show_ids=show_ids)
            if rows:
                last_a, last_d = rows[-1]['alpha_degrees'], rows[-1]['distance_to_O']

            fps = 1/(time.time()-prev_t+1e-9); prev_t = time.time()
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
            import traceback
            traceback.print_exc()
            break

    cap.release(); fm.close(); cv2.destroyAllWindows()
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
        process_frame(frame, tracked, fm, show_mesh=True, show_ids=GLOBAL_CONFIG["show_ids"], vis2d=True)
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
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(ow, x2), min(oh, y2)
            crop = frame[y1:y2, x1:x2]
            mpr = fm.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not mpr.multi_face_landmarks: continue

            wlms = getattr(mpr, 'multi_face_world_landmarks', None)
            wlms0 = wlms[0] if wlms else None
            _, wc_d = build_coords(mpr.multi_face_landmarks[0], wlms0, x1, y1, *crop.shape[:2][::-1])

            fig = go.Figure()

            # 1. Landmark mặt (vùng trên)
            # Mesh xám mờ + text ID (nếu GLOBAL_CONFIG bật)
            upper_ids = [idx for idx in range(468) if idx in wc_d and (idx < 10 or (33 <= idx < 160) or (263 <= idx < 390) or idx in KEY_IDS)]
            pts = np.array([wc_d[idx] for idx in upper_ids])
            
            mode_3d = 'markers+text' if GLOBAL_CONFIG["show_ids"] else 'markers'
            txt_3d  = [str(idx) for idx in upper_ids] if GLOBAL_CONFIG["show_ids"] else None
            
            fig.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2], 
                mode=mode_3d, 
                marker=dict(size=2, color='gray', opacity=0.4), 
                text=txt_3d,
                textposition="top center",
                textfont=dict(size=7, color='black'),
                name='Face Mesh'
            ))

            # 2. Các thành phần hốc mắt
            for eye_idx, eye_ids in enumerate([(163,157,161,154,468),(390,384,388,381,473)]):
                ep = [_pt(wc_d, idx) for idx in eye_ids[:4]]
                if any(p is None for p in ep): continue
                p1, p2, p3, p4 = ep
                # L1, L2 (Cyan)
                fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color='cyan', width=3), name='Hốc mắt' if eye_idx==0 else "", showlegend=eye_idx==0))
                fig.add_trace(go.Scatter3d(x=[p3[0], p4[0]], y=[p3[1], p4[1]], z=[p3[2], p4[2]], mode='lines', line=dict(color='cyan', width=3), showlegend=False))
                
                O,u1,u2 = shortest_common_perpendicular(p1,p2,p3,p4)
                if O is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[O[0]], y=[O[1]], z=[O[2]], 
                        mode='markers+text', 
                        marker=dict(size=6, symbol='x', color='magenta'), 
                        text=["O (Eyeball)"],
                        textposition="top right",
                        name='Tâm ảo O' if eye_idx==0 else "", 
                        showlegend=eye_idx==0
                    ))
                    ox, oy = get_local_axes(u1, u2)
                    sc = 0.03
                    # Ox (Green), Oy (Blue)
                    fig.add_trace(go.Scatter3d(x=[O[0], O[0]+ox[0]*sc], y=[O[1], O[1]+ox[1]*sc], z=[O[2], O[2]+ox[2]*sc], mode='lines', line=dict(color='green', width=5), name='Trục Ox (Ngang)' if eye_idx==0 else "", showlegend=eye_idx==0))
                    fig.add_trace(go.Scatter3d(x=[O[0], O[0]+oy[0]*sc], y=[O[1], O[1]+oy[1]*sc], z=[O[2], O[2]+oy[2]*sc], mode='lines', line=dict(color='blue', width=5), name='Trục Oy (Dọc)' if eye_idx==0 else "", showlegend=eye_idx==0))

                    # Pupil projection
                    pupil = _pt(wc_d, eye_ids[4])
                    if pupil is not None:
                        fig.add_trace(go.Scatter3d(
                            x=[pupil[0]], y=[pupil[1]], z=[pupil[2]], 
                            mode='markers+text', 
                            marker=dict(size=5, color='darkblue'), 
                            text=[str(eye_ids[4])],
                            textposition="bottom center",
                            name='Đồng tử' if eye_idx==0 else "", 
                            showlegend=eye_idx==0
                        ))
                        V = pupil - O
                        proj_pt = O + np.dot(V, ox)*ox + np.dot(V, oy)*oy
                        fig.add_trace(go.Scatter3d(x=[pupil[0], proj_pt[0]], y=[pupil[1], proj_pt[1]], z=[pupil[2], proj_pt[2]], mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=False))

            # 3. Face Basis (168) & Final Gaze
            p168 = _pt(wc_d, 168)
            alpha, d, _, Vf = calculate_gaze(wc_d)
            if p168 is not None:
                v_l, v_r = wc_d[331]-p168, wc_d[102]-p168
                nf = np.cross(v_l, v_r); nf /= np.linalg.norm(nf)
                if nf[2] > 0: nf = -nf
                rf = wc_d[102]-wc_d[331]; rf -= np.dot(rf, nf)*nf; rf /= np.linalg.norm(rf)
                uf = np.cross(nf, rf)
                if uf[1] < 0: uf = -uf
                
                s_f = 0.08
                for idx, (v, c, n) in enumerate([(nf, 'black', 'Pháp tuyến mặt'), (rf, 'orange', 'Trục ngang mặt'), (uf, 'purple', 'Trục dọc mặt')]):
                    fig.add_trace(go.Scatter3d(x=[p168[0], p168[0]+v[0]*s_f], y=[p168[1], p168[1]+v[1]*s_f], z=[p168[2], p168[2]+v[2]*s_f], mode='lines', line=dict(color=c, width=6), name=n))
                if Vf is not None:
                    fig.add_trace(go.Scatter3d(x=[p168[0], p168[0]+Vf[0]*0.18], y=[p168[1], p168[1]+Vf[1]*0.18], z=[p168[2], p168[2]+Vf[2]*0.18], mode='lines', line=dict(color='red', width=10), name='HƯỚNG NHÌN CUỐI'))

            # Layout config
            # Camera Up: Y+ (Up). View from Front (Z-).
            camera = dict(eye=dict(x=0, y=0, z=-1.5), up=dict(x=0, y=1, z=0))
            
            st_alpha = f"{alpha:.1f}" if alpha is not None else "--"
            st_d = f"{d:.4f}" if d is not None else "--"
            
            fig.update_layout(
                scene=dict(
                    camera=camera,
                    xaxis=dict(range=[pts[:,0].min()-0.05, pts[:,0].max()+0.05], autorange=False, title='X (Ngang)'),
                    yaxis=dict(range=[pts[:,1].min()-0.05, pts[:,1].max()+0.05], autorange=False, title='Y (Dọc)'),
                    zaxis=dict(range=[pts[:,2].min()-0.05, pts[:,2].max()+0.05], autorange=False, title='Z (Sâu)'),
                    aspectmode='cube'
                ),
                title=f"3D Gaze Analysis: {path.name} | Alpha: {st_alpha}° | D: {st_d}",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            out_html = OUTPUT_VIS_3D / f"3d_{path.stem}.html"
            fig.write_html(str(out_html))
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
    
    if v_files:
        print(f"  [*] Tìm thấy {len(v_files)} video trong {INPUT_VIDEO}")
        to_process = v_files
    else:
        print(f"  [*] Không tìm thấy video trong {INPUT_VIDEO}. Mở hộp thoại chọn file...")
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        vpath = filedialog.askopenfilename(title="Chọn file video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        root.destroy()
        if not vpath: print("  [!] Không có file nào được chọn."); return
        to_process = [Path(vpath)]

    dev = get_device()
    print(f"  [*] Khởi tạo Face Detection trên: {dev.upper()}")
    yolo = YOLO(str(YOLO_FACE_MODEL)).to(dev)
    print("  [*] Khởi tạo Landmark Extractor trên: CPU (MediaPipe)")
    fm   = make_face_mesh(static=False)
    
    print("\n  [?] Chọn chế độ xử lý:")
    print("      1. Fast Mode (720p, High FPS, Có Preview)")
    print("      2. Ultra-Safe (Low-res, No-Preview, Ổn định cao)")
    safe_choice = input("      Lựa chọn (1 hoặc 2): ").strip()
    
    use_preview = (safe_choice != "2")
    yolo_imgsz = 320 if safe_choice == "2" else 640
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
        last_v = None # State cho smoothing
        v_start_t = time.time()
        v_prev_t = v_start_t
        proc_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None: break
                curr_f += 1
                
                if vw_orig != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))

                yolo_step = 5 if safe_choice == "2" else 3
                if curr_f % yolo_step == 0 or not last_tracked:
                    res = yolo(frame, conf=GLOBAL_CONFIG["yolo_conf"], imgsz=yolo_imgsz, verbose=False)[0]
                    last_tracked = tracker.update(res.boxes)
                
                if last_tracked:
                    rows, _ = process_frame(frame, last_tracked, fm, tracker=tracker, show_mesh=show_mesh, show_ids=show_ids)

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
        print(f"\n    [✓] Xong: {out_path.name}")
        
    fm.close(); cv2.destroyAllWindows()
    print(f"\n[DONE] Hoàn tất xử lý video.\n")

def run_toggle_ids():
    GLOBAL_CONFIG["show_ids"] = not GLOBAL_CONFIG["show_ids"]
    print(f"  [CONFIG] Hiển thị Landmark IDs: {'BẬT' if GLOBAL_CONFIG['show_ids'] else 'TẮT'}")

def run_toggle_gaze_mode():
    GLOBAL_CONFIG["use_eye_gaze"] = not GLOBAL_CONFIG["use_eye_gaze"]
    mode = "Mặt + Mắt" if GLOBAL_CONFIG["use_eye_gaze"] else "Chỉ hướng Mặt"
    print(f"  [CONFIG] Chế độ tính toán: {mode}")

def run_toggle_device():
    import torch
    if GLOBAL_CONFIG["force_device"] == "cpu":
        if torch.cuda.is_available():
            GLOBAL_CONFIG["force_device"] = "cuda"
            print(f"  [CONFIG] Đã chuyển sang: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("\n  [!] KHÔNG THỂ CHUYỂN SANG GPU:")
            if "cpu" in torch.__version__:
                print("      -> Nguyên nhân: Bạn đang dùng phiên bản Torch-CPU.")
                print("      -> Khắc phục: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("      -> Nguyên nhân: CUDA không khả dụng hoặc driver lỗi.")
            return
    else:
        GLOBAL_CONFIG["force_device"] = "cpu"
        print(f"  [CONFIG] Đã chuyển sang: CPU")

def run_settings_menu():
    while True:
        print("\n" + "="*40)
        print("      CÀI ĐẶT THAM SỐ HỆ THỐNG")
        print("="*40)
        print(f"  1. Smooth Alpha [{GLOBAL_CONFIG['smooth_alpha']}] (0.1 - 0.9)")
        print(f"  2. Độ nhạy Ngang S_H [{GLOBAL_CONFIG['S_H']}]")
        print(f"  3. Độ nhạy Dọc S_V [{GLOBAL_CONFIG['S_V']}]")
        print(f"  4. YOLO Confidence [{GLOBAL_CONFIG['yolo_conf']}] (0.1 - 0.9)")
        print("  0. Quay lại Menu chính")
        
        c = input("\nChọn mục cần chỉnh: ").strip()
        if c == '0': break
        try:
            if c == '1':
                v = float(input("Nhập Smooth Alpha mới: "))
                if 0 < v < 1: GLOBAL_CONFIG['smooth_alpha'] = v
            elif c == '2':
                GLOBAL_CONFIG['S_H'] = float(input("Nhập S_H mới: "))
            elif c == '3':
                GLOBAL_CONFIG['S_V'] = float(input("Nhập S_V mới: "))
            elif c == '4':
                v = float(input("Nhập YOLO Confidence mới: "))
                if 0 < v < 1: GLOBAL_CONFIG['yolo_conf'] = v
            print("  [OK] Đã cập nhật.")
        except ValueError:
            print("  [!] Giá trị không hợp lệ.")

# ------------------------------------------------------------------ #
#  MENU                                                                #
# ------------------------------------------------------------------ #
def get_menu():
    st_ids = "BẬT" if GLOBAL_CONFIG["show_ids"] else "TẮT"
    st_gaze = "Mặt+Mắt" if GLOBAL_CONFIG["use_eye_gaze"] else "Chỉ Mặt"
    dev = GLOBAL_CONFIG["force_device"].upper()
    return f"""
╔══════════════════════════════════════════╗
║    GEOMETRIC GAZE ESTIMATION             ║
╠══════════════════════════════════════════╣
║  1  Batch  (ảnh trong input/)            ║
║  2  Webcam (real-time camera)            ║
║  3  Vis 2D (debug hình học trên ảnh)     ║
║  4  Vis 3D (mô hình 3D tương tác)        ║
║  5  Video  (xử lý file video)            ║
║  6  Cấu hình Landmark IDs [{st_ids}]         ║
║  7  Chế độ Gaze [{st_gaze}]            ║
║  8  Thiết bị [{dev}]                     ║
║  9  Cài đặt tham số nâng cao             ║
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
    '9': run_settings_menu
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
