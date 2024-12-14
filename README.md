# yolov8
!pip install ultralytics
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import urllib.request

# 모델 로드
model = YOLO('yolov8x.pt')  # 'n'은 nano 모델을 의미합니다. 더 큰 모델을 원하면 's', 'm', 'l', 'x'로 변경 가능합니다.

# 이미지 경로 (구글 드라이브를 사용하는 경우 경로를 적절히 수정하세요)
img_path = '/content/자동차 경로.jpg?auth=745d3fef479bac83fb073e1fcd5da038c3103f2d97694f0e27b06477e4798667'


import urllib.request

img_url = "https://www.chosun.com/resizer/v2/GM2JXZGOP5DPNOFN6DSS32ECEE.jpg?auth=745d3fef479bac83fb073e1fcd5da038c3103f2d97694f0e27b06477e4798667&width=616"
img_path = '/content/downloaded_image.jpg'
urllib.request.urlretrieve(img_url, img_path)

# 이미지에서 객체 감지
results = model(img_path, conf=0.25, imgsz=1280)

# 결과 시각화
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
car_count = 0
traffic_light_count = 0
person_count = 0

for r in results:
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # 바운딩 박스 좌표
        c = box.cls
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(img, f'{model.names[int(c)]}', (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        class_name = model.names[int(c)]  # class_name 정의
        if class_name == 'car':
           car_count += 1
        elif class_name == 'traffic light':
             traffic_light_count += 1
        elif class_name == 'person':
             person_count += 1

print(f"자동차 수: {car_count}")
print(f"신호등 수: {traffic_light_count}")
print(f"사람 수: {person_count}")

# 결과 이미지 표시
cv2_imshow(img)
