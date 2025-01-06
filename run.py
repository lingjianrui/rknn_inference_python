import cv2
import numpy as np
from rknn.api import RKNN

# 配置路径
MODEL_PATH = '../data/best.rknn'  # RKNN 模型路径
IMAGE_PATH = './aaa.jpg'         # 输入图片路径
INPUT_SIZE = (640, 640)            # YOLOv8 模型输入尺寸
CONF_THRESHOLD = 0.6              # 置信度阈值
OUTPUT_PATH = './detection_result.jpg'  # 输出图片路径
nms_threshold = 0.4

# 获取图片的原始尺寸
original_image = cv2.imread(IMAGE_PATH)
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = original_image.shape[:2]

# 1. 图片预处理
def preprocess_image(image_path, input_size):
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 调整为模型输入尺寸
    image_resized = cv2.resize(image, input_size)

    # 数据格式调整为 NHWC
    image_data = np.expand_dims(image_resized, axis=0)

    return image_data.astype(np.float32)

# 2. Sigmoid 归一化函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

# 4. 推理后处理
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

# 5. 绘制边界框并保存图片
def draw_boxes(image, results, output_path):
    for result in results:
        class_id = result['class_id']
        score = result['score']
        bbox = result['bbox']
        # 绘制边界框
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # 绘制类别ID和分数
        cv2.putText(image, f"Class {class_id} {score:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 保存图片
    cv2.imwrite(output_path, image)

def main():
    # 初始化 RKNN 对象
    rknn = RKNN()

    # 加载 RKNN 模型
    print('Loading RKNN model...')
    if rknn.load_rknn(MODEL_PATH) != 0:
        print('Load RKNN model failed!')
        return

    # 初始化运行时环境
    print('Initializing runtime environment...')
    if rknn.init_runtime(target='rk3588') != 0:
        print('Init runtime environment failed!')
        return

    # 图片预处理
    print('Preprocessing image...')
    image_data = preprocess_image(IMAGE_PATH, INPUT_SIZE)
    print('Image data shape:', image_data.shape)

    # 推理
    print('Running inference...')
    outputs = rknn.inference(inputs=[image_data])
    if isinstance(outputs, list):
        outputs = np.array(outputs)
    # 将输出转换成[8400,7]
    outputs = outputs[0][0].T
    print(f"Output shape: {outputs.shape}")
    print('Raw Outputs:', outputs)
    print(f"Output shape: {outputs.shape}")
    print(f"Sample data: {outputs[0][:10]}")

    # 后处理
    print('Postprocessing results...')
    results = postprocess(outputs, CONF_THRESHOLD, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, INPUT_SIZE)
    # 应用非极大值抑制
    indices = nms([r['bbox'] for r in results], [r['score'] for r in results], nms_threshold)
    results = [results[i] for i in indices]
    print('Detection results:')
    for result in results:
        print(f"Class ID: {result['class_id']}, Score: {result['score']}, BBox: {result['bbox']}")

    # 绘制边界框并保存图片
    if results:
        print('Drawing bounding boxes and saving image...')
        image = cv2.imread(IMAGE_PATH)
        draw_boxes(image, results, OUTPUT_PATH)

    # 释放资源
    print('Releasing RKNN resources...')
    rknn.release()

# 执行主函数
if __name__ == '__main__':
    main()
