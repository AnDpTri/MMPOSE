import numpy as np

"""
BẢN SAO LƯU LOGIC TÍNH TOÁN HƯỚNG NHÌN (GAZE) VÀ HƯỚNG MẶT (FACE ORIENTATION)
Dự án: Geometric Gaze Estimation
-------------------------------------------------------------------------
Tài liệu này chứa toàn bộ các hàm cốt lõi để tính toán vector hướng nhìn 
dựa trên các điểm mốc (landmarks) từ MediaPipe World Landmarks.
"""

def shortest_common_perpendicular(P1, P2, P3, P4):
    """
    TÌM GIAO ĐIỂM ẢO (O) CỦA CÁC TRỤC MẮT
    --------------------------------------
    Mỗi mắt được giả định có 2 trục (L1: đi qua Landmark 163-157 và L2: 161-154).
    Giao điểm ảo O được định nghĩa là trung điểm của đoạn vuông góc chung 
    ngắn nhất giữa hai đường thẳng L1 và L2 này.
    
    Tham số:
        P1, P2: Hai điểm định nghĩa đường thẳng L1 (Vành mắt trên/dưới)
        P3, P4: Hai điểm định nghĩa đường thẳng L2 (Vành mắt trái/phải)
    Trả về:
        O: Tọa độ tâm mắt ảo (3D)
        u1, u2: Vector chỉ phương đã chuẩn hóa của L1 và L2
    """
    # 1. Tính vector chỉ phương
    u1 = P2 - P1; n1 = np.linalg.norm(u1)
    if n1 < 1e-6: return None, None, None
    u1 /= n1
    
    u2 = P4 - P3; n2 = np.linalg.norm(u2)
    if n2 < 1e-6: return None, None, None
    u2 /= n2
    
    # 2. Giải hệ phương trình tìm đoạn vuông góc chung
    dp = P1 - P3
    dot12 = np.dot(u1, u2)
    dot1p = np.dot(u1, dp)
    dot2p = np.dot(u2, dp)
    
    det = 1.0 - dot12**2
    if det < 1e-7: # Trường hợp 2 đường thẳng song song
        return (P1 + P3) / 2, u1, u2

    s = (dot12 * dot2p - dot1p) / det
    t = (dot2p - dot12 * dot1p) / det
    
    # C1, C2 là hai điểm gần nhau nhất trên L1 và L2
    C1 = P1 + s * u1
    C2 = P3 + t * u2
    
    # Tâm O là trung điểm của C1C2
    O = (C1 + C2) / 2
    return O, u1, u2

def get_local_axes(u1, u2):
    """
    THIẾT LẬP HỆ TỌA ĐỘ ĐỊA PHƯƠNG TẠI MẮT (Ox, Oy)
    ----------------------------------------------
    Từ hai trục mắt u1 và u2, ta tạo ra hệ tọa độ vuông góc để chiếu con ngươi.
    Ox (Ngang) và Oy (Dọc) được xác định dựa trên các đường phân giác.
    
    Logic phân loại:
        - b1 = u1 + u2; b2 = u1 - u2
        - Trục nào có thành phần X (ngang) lớn hơn sẽ là Ox.
        - Trục còn lại là Oy.
    """
    b1 = u1 + u2; b2 = u1 - u2
    n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
    b1 = b1/n1 if n1 > 1e-6 else u1
    b2 = b2/n2 if n2 > 1e-6 else u2
    
    # MediaPipe World: X+ là Right (nhìn từ camera), Y+ là Up
    if abs(b1[0]) > abs(b1[1]): # Ưu tiên thành phần X cho trục Ngang
        ox, oy = b1, b2
    else: 
        ox, oy = b2, b1
        
    # Chuẩn hóa hướng để Ox luôn hướng sang phải (theo chủ thể) và Oy hướng lên trên
    if ox[0] > 0: ox = -ox
    if oy[1] < 0: oy = -oy
    
    return ox, oy

def process_eye(P1, P2, P3, P4, pupil):
    """
    TÍNH TOÁN TỌA ĐỘ CỰC CỦA CON NGƯƠI (Alpha, d)
    --------------------------------------------
    Tính toán xem con ngươi (pupil) đang lệch bao nhiêu so với tâm mắt O 
    trên hệ tọa độ địa phương (Ox, Oy).
    
    Trả về:
        alpha: Góc lệch (độ)
        d: Độ dài hình chiếu (độ lệch tâm)
    """
    # 1. Tìm O và các trục địa phương
    O, u1, u2 = shortest_common_perpendicular(P1, P2, P3, P4)
    if O is None: return None, None
    Ox, Oy = get_local_axes(u1, u2)
    
    # 2. Chiếu vị trí con ngươi xuống mặt phẳng mắt
    V = pupil - O
    x0 = np.dot(V, Ox) # Tọa độ ngang
    y0 = np.dot(V, Oy) # Tọa độ dọc
    
    # 3. Chuyển sang Alpha (góc) và d (khoảng cách)
    alpha = np.degrees(np.arctan2(y0, x0))
    d = np.linalg.norm([x0, y0])
    return alpha, d

def calculate_gaze_logic(world_landmarks, use_eye=True):
    """
    LOGIC TỔNG HỢP: HƯỚNG MẶT + HƯỚNG MẮT
    -------------------------------------
    Kết hợp hướng của khuôn mặt và độ lệch của con ngươi để ra vector Gaze cuối cùng.
    
    Các bước:
    1. Tính Alpha (góc) và D (độ lệch) trung bình của 2 mắt.
    2. Xác định mặt phẳng khuôn mặt (V_face) từ 3 điểm: Gốc mũi (168), 2 hốc mắt (102, 331).
    3. Tạo hệ trục Face (Front, Right, Up).
    4. Bẻ hướng vector V_face dựa trên Alpha, D và hệ số S_H, S_V (độ nhạy).
    """
    # ... Giả định world_landmarks đã được trích xuất tọa độ ...
    
    # (A) TRỤC KHUÔN MẶT (Orientation)
    # V_face: Pháp tuyến của mặt phẳng mặt, hướng về phía trước (Z âm)
    # Rf: Trục ngang của mặt (Right)
    # Uf: Trục dọc của mặt (Up) = V_face x Rf
    
    # (B) TÍNH GAZE (Nếu use_eye=True)
    # Công thức bẻ hướng:
    # V_final = V_face + d_avg * (cos(alpha)*S_H*Rf + sin(alpha)*S_V*Uf)
    # Trong đó S_H=18, S_V=25 là các hệ số nhạy cho trục Ngang và Dọc.
    
    # (C) SMOOTHING (EMA - Nếu chạy trong vòng lặp)
    # V_smooth = alpha * V_new + (1 - alpha) * V_old
    
    pass # Hàm này trong script chính thực hiện các phép tính vector tương tự

if __name__ == "__main__":
    print("Đây là script backup logic hình học. Vui lòng tham khảo code bên trong.")
