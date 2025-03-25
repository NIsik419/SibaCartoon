import cv2
import numpy as np

def color_quantization(img, k=8):
    data = img.reshape((-1, 3)).astype(np.float32)
    _, label, center = cv2.kmeans(data, k, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                   10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(img.shape)

def cartoon_render(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {img_path}")
        return

    # 1. 컬러 단순화 (색상 퀀타이징)
    quantized = color_quantization(img, k=11)

    # 2. 엣지 검출 (Canny Edge)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # 3. 윤곽선 반전 → 흰 배경에 검정 선
    edges_inv = cv2.bitwise_not(edges)

    # 4. 윤곽선을 3채널로 변환
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # 5. 컬러와 윤곽선 결합 (bitwise_and)
    cartoon = cv2.bitwise_and(quantized, edges_colored)

    # 결과 저장
    cv2.imwrite(save_path, cartoon)


cartoon_render("images/good_example.webp", "results/cartoon_good.jpg")
cartoon_render("images/bad_example.jpg", "results/cartoon_bad.jpg")
