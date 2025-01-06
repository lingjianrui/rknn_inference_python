import cv2
import numpy as np
from rknn.api import RKNN
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time

# 配置路径
MODEL_PATH = '../data/best.rknn'  # RKNN 模型路径
INPUT_SIZE = (640, 640)            # YOLOv8 模型输入尺寸
CONF_THRESHOLD = 0.6              # 置信度阈值
OUTPUT_VIDEO_PATH = './detection_result'  # 输出视频路径

# 1. 图片预处理
def preprocess_image(image, input_size):
    # 调整为模型输入尺寸
    image_resized = cv2.resize(image, input_size)

    # 数据格式调整为 NHWC
    image_data = np.expand_dims(image_resized, axis=0)

    return image_data.astype(np.float32)

# 2. Sigmoid 归一化函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. 推理后处理
def postprocess(outputs, conf_threshold, original_width, original_height, input_size):
    results = []
    detections = outputs
    for detection in detections:
        x_center, y_center, width, height, confidence, class1, class2 = detection
        # 对置信度进行Sigmoid归一化
        confidence = sigmoid(confidence)
        if confidence > conf_threshold:
            # 将中心点和宽高从输入尺寸映射到原始尺寸
            x1 = int((x_center - width / 2) * original_width / input_size[0])
            y1 = int((y_center - height / 2) * original_height / input_size[1])
            x2 = int((x_center + width / 2) * original_width / input_size[0])
            y2 = int((y_center + height / 2) * original_height / input_size[1])
            results.append({
                "class_id": int(class1),  # 假设class1是类别ID
                "score": float(confidence),
                "bbox": [x1, y1, x2, y2]
            })
    return results

# 4. 绘制边界框并保存图片
def draw_boxes(image, results):
    for result in results:
        class_id = result['class_id']
        score = result['score']
        bbox = result['bbox']
        # 绘制边界框
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # 绘制类别ID和分数
        cv2.putText(image, f"Class {class_id} {score:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 3. 非极大值抑制
def nms(boxes, scores, nms_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]
    return keep


# 添加一个结果对象类来传递数据
class DetectionResult:
    def __init__(self, frame, frame_count):
        self.frame = frame
        self.frame_count = frame_count

def save_frame(output_dir, result):
    """保存帧的线程函数"""
    output_path = os.path.join(output_dir, f'detection_{result.frame_count:04d}.jpg')
    cv2.imwrite(output_path, result.frame)
    print(f'Saved frame to {output_path}')

def main():
    # 创建保存结果的目录
    output_dir = 'detection_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')

    # 初始化 RKNN
    rknn = RKNN()
    print('Loading RKNN model...')
    if rknn.load_rknn(MODEL_PATH) != 0:
        print('Load RKNN model failed!')
        return

    print('Initializing runtime environment...')
    if rknn.init_runtime(target='rk3588') != 0:
        print('Init runtime environment failed!')
        return

    # 打开摄像头
    cap = cv2.VideoCapture(74)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    # 创建线程池
    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_count = 0
        running = True
        
        # 初始化时间计数器
        process_times = []
        start_time = time.time()

        while running:
            try:
                # 开始计时
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    continue

                # 图片预处理
                image_data = preprocess_image(frame, INPUT_SIZE)

                # 推理
                outputs = rknn.inference(inputs=[image_data])
                if isinstance(outputs, list):
                    outputs = np.array(outputs)
                outputs = outputs[0][0].T

                # 后处理
                results = postprocess(outputs, CONF_THRESHOLD, 640, 640, INPUT_SIZE)
                
                # 应用非极大值抑制
                indices = nms([r['bbox'] for r in results], [r['score'] for r in results], nms_threshold)
                results = [results[i] for i in indices]
                print('Detection results:')
                for result in results:
                    print(f"Class ID: {result['class_id']}, Score: {result['score']}, BBox: {result['bbox']}")

                # 绘制边界框并保存图片
                if results:
                    print('Drawing bounding boxes and saving image...')
                    draw_boxes(frame, results)

                # 创建结果对象并提交保存任务
                detection_result = DetectionResult(frame.copy(), frame_count)
                executor.submit(save_frame, output_dir, detection_result)

                # 计算这一帧的处理时间
                frame_time = time.time() - frame_start
                process_times.append(frame_time)
                
                # 计算实时FPS
                current_fps = 1.0 / frame_time
                
                # 每10帧显示一次FPS
                if frame_count % 10 == 0:
                    avg_fps = 1.0 / (sum(process_times[-10:]) / len(process_times[-10:]))
                    print(f"Current FPS: {current_fps:.2f}, Average FPS (last 10 frames): {avg_fps:.2f}")

                frame_count += 1

                # 可选：限制帧数
                if frame_count >= 100:  # 比如只保存100帧
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                break

        # 计算总体统计信息
        total_time = time.time() - start_time
        avg_frame_time = sum(process_times) / len(process_times)
        avg_fps = 1.0 / avg_frame_time
        
        print("\n=== Performance Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average processing time per frame: {avg_frame_time*1000:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Min FPS: {1.0/max(process_times):.2f}")
        print(f"Max FPS: {1.0/min(process_times):.2f}")

        # 释放资源
        print('\nReleasing resources...')
        rknn.release()
        cap.release()

if __name__ == '__main__':
    main()
