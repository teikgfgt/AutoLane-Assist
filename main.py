import cv2
import numpy as np
import serial
import time
import signal
import sys
from ultralytics import YOLO

# --- 1. Mở cổng Serial với Arduino ---
def connect_arduino(port, baudrate, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print("✅ Arduino kết nối thành công.")
            return arduino
        except Exception as e:
            print(f"❌ Thử {attempt+1}/{max_attempts} thất bại: {e}")
            time.sleep(1)
    print("❌ Không thể kết nối Arduino.")
    return None

arduino = connect_arduino('COM9', 9600)

# --- 2. Hàm gửi lệnh dừng ---
def send_stop_command():
    if not arduino or not arduino.is_open:
        print("❌ Cổng Serial không mở, không thể gửi lệnh dừng.")
        return
    try:
        arduino.write('S'.encode('utf-8'))
        arduino.flush()
        print("📤 Gửi lệnh dừng: S")
    except Exception as e:
        print("❌ Lỗi gửi lệnh dừng:", e)

# --- 3. Xử lý khi thoát chương trình ---
def signal_handler(sig, frame):
    print("⏹ Đang thoát chương trình...")
    send_stop_command()
    if arduino and arduino.is_open:
        arduino.close()
        print("Đóng Arduino.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- 4. Load model YOLO ---
model = YOLO("chi.pt")  # Thay đường dẫn model của bạn

# --- 5. Hàm gửi lệnh chỉ khi thay đổi ---
last_command = None
def send_to_arduino(cmd):
    global last_command
    if not arduino or not arduino.is_open:
        print("❌ Cổng Serial không mở, không thể gửi lệnh.")
        return
    if cmd != last_command:
        try:
            arduino.write(cmd.encode('utf-8'))
            arduino.flush()
            print(f"📤 Gửi lệnh: {cmd}")
            last_command = cmd
        except Exception as e:
            print("❌ Lỗi gửi lệnh:", e)

# --- 6. Tính độ lệch và quyết định lệnh ---
def calculate_deviation(mask, frame_w):
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        return None, None
    c = max(ctrs, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, None
    x, y, w, h = cv2.boundingRect(c)
    lane_c = x + w // 2
    mid = frame_w // 2
    dev = lane_c - mid
    if dev < -10:  # Quay lại ngưỡng cũ
        return abs(dev), 'L'
    elif dev > 10:
        return abs(dev), 'R'
    else:
        return abs(dev), 'S'

# --- 7. Vẽ lên frame ---
def draw_info(frame, mask, dev, cmd, bbox):
    overlay = frame.copy()
    overlay[mask==255] = (0,255,0)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    if bbox:
        x,y,w,h = bbox
        lc = x + w//2; fc = frame.shape[1]//2
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.line(frame, (lc,y), (lc,y+h), (255,0,0), 2)
        cv2.line(frame, (fc,0), (fc,frame.shape[0]), (0,0,255), 2)
        cv2.putText(frame, f"Dev: {dev}px  Cmd: {cmd}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

# --- 8. Xử lý video ---
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Không thể mở video.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⏹ Video kết thúc.")
                send_to_arduino('S')
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb)

            # Tạo mask và tìm bounding box lớn nhất
            mask = np.zeros((h, w), dtype=np.uint8)
            for r in results:
                if hasattr(r, 'masks') and r.masks is not None:
                    for i, box in enumerate(r.boxes):
                        if box.cls == 0:
                            seg = r.masks[i].data[0].cpu().numpy()
                            seg = (seg*255).astype(np.uint8)
                            seg = cv2.resize(seg, (w, h))
                            mask = cv2.bitwise_or(mask, seg)

            # Tính deviation và gửi lệnh
            dev, cmd = calculate_deviation(mask, w)
            if cmd is not None:
                send_to_arduino(cmd)
                frame = draw_info(frame, mask, dev, cmd, (lambda c=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]: 
                                                           cv2.boundingRect(max(c, key=cv2.contourArea)))())
            else:
                send_to_arduino('S')
                frame = draw_info(frame, mask, 0, 'S', None)

            cv2.imshow("Lane Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹ Thoát do người dùng nhấn 'q'.")
                send_to_arduino('S')
                break

    except KeyboardInterrupt:
        print("⏹ Đang thoát do Ctrl+C...")
    except Exception as e:
        print(f"❌ Lỗi xử lý video: {e}")
    finally:
        send_stop_command()
        cap.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()
            print("Đóng Arduino.")

# --- 9. Chạy ---
if __name__ == "__main__":
    try:
        video_file = "teik.mp4"  # Thay bằng đường dẫn của bạn
        process_video(video_file)
    except Exception as e:
        print(f"❌ Lỗi chương trình: {e}")
        send_stop_command()
        if arduino and arduino.is_open:
            arduino.close()
            print("Đóng Arduino.")