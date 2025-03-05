import cv2
from ultralytics import YOLO
import time
from pymavlink import mavutil

# 初始化YOLOv8-nano模型
model = YOLO('yolov8n.pt')

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，根据实际情况修改

# 连接无人机（示例使用串口连接）
# 实际使用时需要修改端口和波特率
# connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)

# 检测参数配置
min_width = 100    # 最小报警宽度（像素）
min_height = 100   # 最小报警高度（像素）
log_file = 'detection_log.txt'  # 日志文件路径

def get_drone_position():
    """获取无人机当前位置信息（需要根据实际MAVLink协议调整）"""
    try:
        # 示例获取位置信息，实际需要根据MAVLink消息格式解析
        # msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        # return (msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1e3)
        return (0.0, 0.0, 0.0)  # 示例返回值
    except Exception as e:
        print(f"获取位置失败: {e}")
        return None

def write_log(data):
    """写入日志文件"""
    with open(log_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {data}\n")

# 主循环
last_time = time.time()
while cap.isOpened():
    current_time = time.time()
  
    if current_time - last_time < 1.0:
        continue
    last_time = current_time
    
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break
    
    # 执行YOLOv8检测
    results = model(frame)
    
    # 处理检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取检测框信息
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            
            # 记录到日志文件
            log_data = f"{cls_name} ({width:.1f}x{height:.1f}) @ [{xyxy[0]:.1f}, {xyxy[1]:.1f}]"
            write_log(log_data)
            
            # 检测是否超过设定大小
            if width > min_width and height > min_height:
                position = get_drone_position()
                if position:
                    geo_log = f"ALERT: {cls_name} 超过尺寸 @ 坐标: {position}"
                    write_log(geo_log)
                    print(geo_log)  # 控制台输出提示

# 释放资源
cap.release()
print("检测程序已停止")
