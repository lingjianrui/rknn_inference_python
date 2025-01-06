import cv2
import numpy as np
from rknn.api import RKNN
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

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

def display_frame(result, display_queue):
    """显示帧的线程函数"""
    cv2.imshow('Detection Result', result.frame)
    display_queue.put(True)  # 通知主线程已显示

def main():
    # 创建保存结果的目录
    output_dir = 'detection_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')

    # 初始化显示窗口
    cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Result', 1280, 720)

    # 创建线程池
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 创建一个队列用于显示同步
        display_queue = Queue()

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

        frame_count = 0
        running = True

        while running:
            try:
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

                # 绘制边界框
                draw_boxes(frame, results)

                # 创建结果对象
                detection_result = DetectionResult(frame.copy(), frame_count)

                # 提交保存任务
                executor.submit(save_frame, output_dir, detection_result)
                
                # 提交显示任务
                executor.submit(display_frame, detection_result, display_queue)

                # 等待显示完成
                display_queue.get()

                # 检查键盘输入
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

                frame_count += 1

                # 可选：限制帧数
                if frame_count >= 100:
                    running = False
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                running = False
                break

        # 释放资源
        print('Releasing resources...')
        rknn.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
