import numpy as np
import matplotlib.pyplot as plt
import cv2


def kmeans(pixels, iterations, k=3):
    global labels
    centroids = np.array([[255, 0, 0],  # Đỏ
                          [0, 255, 0],  # Xanh lá
                          [128, 128, 128]])  # Màu xám

    for _ in range(iterations):
        distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            if np.any(labels == i):
                centroids[i] = pixels[labels == i].mean(axis=0)

    quantized_image = centroids[labels].reshape(-1, 3)

    return quantized_image.astype(np.uint8), labels, centroids


def detect_ripe_fruits(image, iterations):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = rgb_image.reshape((-1, 3)).astype(np.float32)

    # Sử dụng hàm kmeans với số vòng lặp
    quantized_image, labels, centers = kmeans(pixel_values, iterations=iterations)

    segmented_image = quantized_image.reshape(rgb_image.shape)

    # Tạo mặt nạ cho màu đỏ với giới hạn rộng hơn
    red_mask = cv2.inRange(segmented_image, (120, 0, 0), (255, 120, 120))

    # Tìm đường viền trong mặt nạ đỏ
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hàm loại bỏ các đường viền nhỏ
    def remove_small_boxes(contours, min_area=1000):
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                filtered_contours.append(cnt)
        return filtered_contours

    contours = remove_small_boxes(contours, min_area=1500)

    # Hàm hợp nhất các hộp gần nhau
    def merge_nearby_boxes(boxes, max_distance=10):
        merged_boxes = []
        for box in boxes:
            x, y, w, h = box
            merged = False
            for i, merged_box in enumerate(merged_boxes):
                mx, my, mw, mh = merged_box
                if (abs(x - mx) < max_distance or
                        abs(x + w - (mx + mw)) < max_distance or
                        abs(y - my) < max_distance or
                        abs(y + h - (my + mh)) < max_distance):
                    x1 = min(x, mx)
                    y1 = min(y, my)
                    x2 = max(x + w, mx + mw)
                    y2 = max(y + h, my + mh)
                    merged_boxes[i] = [x1, y1, x2 - x1, y2 - y1]
                    merged = True
                    break
            if not merged:
                merged_boxes.append([x, y, w, h])
        return merged_boxes

    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    merged_boxes = merge_nearby_boxes(boxes, max_distance=10)

    # Hàm loại bỏ các hộp bên trong
    def remove_inner_boxes(boxes):
        outer_boxes = []
        for box in boxes:
            x, y, w, h = box
            is_inner = False
            for outer_box in outer_boxes:
                ox, oy, ow, oh = outer_box
                if (x >= ox and x + w <= ox + ow and
                        y >= oy and y + h <= oy + oh):
                    is_inner = True
                    break
            if not is_inner:
                outer_boxes.append(box)
        return outer_boxes

    final_boxes = remove_inner_boxes(merged_boxes)

    # Vẽ hình chữ nhật quanh các quả chín
    result_image = rgb_image.copy()
    for box in final_boxes:
        x, y, w, h = box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật quanh quả chín

    return result_image, segmented_image


if __name__ == "__main__":
    image = cv2.imread('picture/RipeFruit2.png')

    # Phát hiện trái chín với 5 vòng lặp
    segmented_result_5, kmeans_result_5 = detect_ripe_fruits(image, iterations=5)

    # Phát hiện trái chín với 10 vòng lặp
    segmented_result_10, kmeans_result_10 = detect_ripe_fruits(image, iterations=10)

    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(kmeans_result_5)
    plt.title('Kết quả phân cụm (5 vòng lặp)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(segmented_result_5)
    plt.title('Kết quả phát hiện trái chín (5 vòng lặp)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(kmeans_result_10)
    plt.title('Kết quả phân cụm (10 vòng lặp)')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(segmented_result_10)
    plt.title('Kết quả phát hiện trái chín (10 vòng lặp)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
