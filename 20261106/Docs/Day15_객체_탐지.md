# 객체 탐지 · IoU · mAP · YOLO(학습, 검증, 추론, 추적)
## 0. BIG PICTURE
### 0-1. 오늘 배울 것
- 이전에 학습한 CNN **분류(Classification)** 모델은 이미지 전체의 클래스를 예측
- YOLO **객체 탐지(Object Detection)**를 활용하여 **객체의 종류·위치·신뢰도**를 예측
### 0-2. YOLO 객체 탐지(Object Detection) 처리 과정
```text
입력 이미지 ─ 이미지 전처리
    ↓
Backbone ─ 이미지의 특징 추출
    ↓
Neck ─ 여러 크기의 특징 결합
    ↓
Head ─ 객체 위치·클래스·신뢰도 예측
    ↓
Confidence 필터링 + NMS ─ 낮은 점수와 중복 후보 정리
    ↓
Results ─ 최종 탐지 결과
```
- 이전 학습에서 **Neck**이 추가된 구조
### 0-3. 탐지 결과의 구성
| 항목 | 설명 | 예시 |
|---|---|---|
| **Box** | 객체가 존재하는 좌표 | `[64, 48, 320, 288]` |
| **Class** | 객체 종류 | person |
| **Confidence** | 예측 점수 | 0.87 |
### 0-4. 탐지 결과를 어떻게 평가하는가?
| 기준 | 역할 | 비교 대상 |
|---|---|---|
| **Confidence** | 낮은 점수의 후보 제거 | 예측 점수 |
| **NMS IoU** | 중복 탐지 제거 | 예측 Box ↔ 예측 Box |
| **평가 IoU** | 위치 정확도 판단 | 예측 Box ↔ 정답 Box |

- **IoU(Intersection over Union)** : 두 박스의 겹침 비율
- 그 외 **평가 지표**
    - **Precision** : 탐지한 객체 중 맞게 탐지한 비율
    - **Recall** : 실제 객체 중 탐지에 성공한 비율
    - **mAP50** : IoU 0.50 기준 클래스별 AP 평균
    - **mAP50-95** : IoU 0.50~0.95 기준의 평균 성능
### 0-5. 객체 탐지 → 객체 추적
- **Class ID** : 객체의 종류를 나타내는 번호
- **Track ID** : 프레임 사이에서 같은 객체를 연결하는 번호
### 0-6. 핵심 정리
[추론·추적]
**이미지 입력 → 객체 탐지 → 중복 제거 → 성능 평가 → 객체 추적 → 결과 저장**

[성능 평가]
**탐지 결과 + 정답 라벨 → IoU·Precision·Recall·mAP**


## 1. 객체 탐지 기초 (Object Detection, Bounding Box, 좌표 변환)
### 1-1. 이미지 분류·객체 탐지·분할
- **분류** Classification : **이미지** 전체의 **클래스** 예측
- **객체 탐지** Detection : **객체별 클래스·위치** 예측
- **객체 분할** Segmentation : 객체·영역의 픽셀 단위 구분

| 구분 | 예측 정보 | 특징 |
|---|---|---|
| **분류** | Class | 이미지 전체를 분류 |
| **탐지** | Class + Box + Confidence | 여러 객체의 종류와 위치 탐지 |
| 분할 | Pixel Mask | 객체의 정확한 영역 구분 |
### 1-2. 객체 탐지의 출력 정보
| 항목 | 의미 | 예시 |
|---|---|---|
| `class` | 객체 종류 | dog |
| `confidence` | 예측 점수 | 0.87 |
| `box` | 객체 위치 | `[64,48,320,288]` |
| `image_id / frame` | 입력 이미지 또는 프레임 | frame 125 |
### 1-3. Bounding Box 좌표
| 형식 | 구성 | 사용 |
|---|---|---|
| xyxy | 왼쪽 위 `(x1,y1)`, 오른쪽 아래 `(x2,y2)` | 박스 그리기, IoU 계산 |
| OpenCV Rect | 왼쪽 위 `(x,y)` + 너비·높이 | OpenCV 사각형 |
| **중심 xywh** | 중심 `(cx,cy)` + 너비·높이 | `boxes.xywh` |
| **정규화**(Normalized) | 이미지 크기로 나눈 0~1 좌표 | YOLO 학습 라벨 |
### 1-4. OpenCV로 Bounding Box 그리기
- cv2.rectangle(img, pt1, pt2, color, thickness=1, lineType=...)
### 1-5 핵심 정리
1) **객체 탐지**(Object Detection)는 객체의 종류뿐 아니라 **위치도 예측**합니다.
2) **Bounding Box**는 xyxy, xywh, 정규화 좌표로 표현할 수 있습니다.
3) **YOLO 라벨**은 원본 이미지 크기로 정규화한 중심 좌표와 너비·높이를 사용합니다.
4) **OpenCV**의 rectangle()과 putText()로 탐지 결과를 시각화합니다.


## 2. 성능 평가 (IoU, NMS, Precision, Recall, mAP)
### 2-1. IoU (Intersection over Union)
- **IoU** : **두 Bounding Box가 얼마나 겹치는지** 나타내는 **지표**
- **박스의 위치와 크기를 평가**하며, **클래스 일치 여부는 별도**로 확인
- IoU = 0 : 겹치지 않음, **IoU = 1 : 완전히 동일**
### 2-2. TP · FP · FN
| 지표 | 의미 | 예시 |
|---|---|---|
| TP (True Positive) | 올바르게 탐지한 객체 | 실제 개를 개로 탐지 |
| FP (False Positive) | 잘못된 탐지 | 배경을 개로 탐지 |
| FN (False Negative) | 탐지하지 못한 정답 객체 | 실제 개를 놓침 |
### 2-3. NMS (Non-Maximum Suppression)
- **NMS** : 같은 객체에 대해 여러 개의 중복 Box가 예측되었을 때, **겹치는 후보를 제거하는 알고리즘**
### 2-4. Precision · Recall
- Precision(정밀도) : 탐지한 객체 중 실제로 맞게 탐지한 비율
- Recall(재현율) : 실제로 존재하는 객체 중 탐지에 성공한 비율
- Precision이 낮으면 잘못된 탐지(FP)가 많다.
- Recall이 낮으면 놓치는 객체(FN)가 많다.
### 2-5. AP · mAP
- AP, mAP : 다양한 Confidence 기준에서 얻은 **Precision–Recall 관계를 요약하는 지표**

| 지표 | 의미 |
|---|---|
| AP | 한 클래스의 Precision–Recall 곡선을 요약 |
| mAP | 클래스별 AP의 평균 |
| mAP50 | 평가 IoU 0.50에서 AP 평균 |
| mAP50-95 | 평가 IoU 0.50~0.95, 0.05 간격의 10개 기준 평균 |

- mAP50이 높고 mAP50-95가 낮다면, 객체의 존재는 잘 찾지만 **Box 위치의 정밀도가 부족**할 가능성
### 2-6. 핵심 정리
- **IoU**는 두 박스의 겹침 정도 계산
- **NMS**는 중복 예측 박스 제거
- **TP·FP·FN**는 올바른 탐지·거짓 탐지·누락 구분
- **Precision**는 탐지 결과가 얼마나 정확한가
- **Recall**는 실제 객체를 얼마나 놓치지 않았는가
- **AP·mAP**는 다양한 기준에서의 탐지 성능 종합 평가

**전체 흐름** : 예측 Box 생성 → Confidence 필터링 → NMS 중복 제거 → 정답과 IoU 비교 → TP·FP·FN 판정 → Precision·Recall·mAP 평가


## 3. 데이터 준비 (라벨링, YOLO 포맷, 데이터 품질 검수)
### 3-1. 라벨링(Labeling)의 중요성
- **라벨링** : **이미지 속 객체의 위치와 클래스**를 **정답 데이터로 기록**하는 작업
- **기본 규칙**
    - 불필요한 배경을 최소화
    - 같은 라벨링 규칙을 적용
    - 누락·중복·가림·잘림에 대한 기준을 정함
    - 개인 판단보다정해진 규칙을 준수
### 3-2. YOLO Label 형식
- class_id cx cy w h

| 항목 | 의미 |
|---|---|
| class_id | 클래스 번호 (0부터 시작) |
| cx | 정규화된 중심 X 좌표 |
| cy | 정규화된 중심 Y 좌표 |
| w | 정규화된 Bounding Box 너비 |
| h | 정규화된 Bounding Box 높이 |
### 3-3. Pixel Box → YOLO Label 변환
```python
# 1. 이미지 크기와 Box 좌표
W, H = 640, 480
x1, y1, x2, y2 = 64, 48, 320, 288

# 2. YOLO 정규화 좌표 변환
cx = ((x1 + x2) / 2) / W
cy = ((y1 + y2) / 2) / H
bw = (x2 - x1) / W
bh = (y2 - y1) / H

# 3. YOLO Label 문자열 생성
line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
print(line)
```
### 3-4. YOLO 데이터셋 폴더 구조
```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```
- **dataset.yaml** : 데이터 경로·클래스 정의
```YAML
path: C:/ai_course/day15/dataset
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: no_helmet
  2: person
```
| 항목 | 설명 |
|---|---|
| path | 데이터셋 루트 경로 |
| train | 학습 이미지 경로 |
| val | 검증 이미지 경로 |
| test | 최종 평가 이미지 경로 (선택) |
| names | Class ID와 클래스 이름 연결 |
### 3-5. 데이터 누수(Data Leakage) 방지
| 상황 | 문제 |
|---|---|
| 원본과 증강 이미지를 다른 Split에 배치 | 거의 동일한 데이터가 학습·검증에 사용됨 |
| 동일 영상의 연속 프레임을 무작위 분할 | 비슷한 장면과 객체가 서로 섞임 |
| 같은 카메라·현장의 연속 촬영 데이터 분할 | 평가 독립성이 저하될 수 있음 |
| Test 결과를 보고 Threshold 반복 수정 | 최종 평가 데이터가 모델 선택에 사용됨 |
### 3-6. Negative Image
- **Negative Image** : 탐지 대상 객체가 없는 이미지
- **탐지할 객체가 없는 장면**도 **학습에 사용**해야 함
### 3-7. 핵심 정리
- **라벨링**(Labeling)은 **객체의 클래스와 위치**를 정답으로 기록한다
- **YOLO Label**은 class_id, cx, cy, w, h 형식으로
- **dataset.yaml**은 데이터 경로 및 클래스 정의
- **탐지 대상이 없는 장면**도 학습에 활용


## 4. YOLO 모델 구조 및 학습 (Backbone·Neck·Head)
### 4-1. YOLO (You Only Look Once)
- **YOLO** : **한 번의 모델 순전파**로 이미지 속 **여러 객체의 위치와 클래스를 예측**하는 **객체 탐지(Object Detection) 모델**
- **One-stage Detector**
    - 입력 이미지를 한 번의 순전파로 처리
    - 다양한 위치와 크기의 객체 후보를 예측
    - 오늘 실습의 NMS 경로에서는 낮은 점수와 중복 후보를 제거
    - 정확도뿐 아니라 추론 지연시간·FPS·메모리도 중요
### 4-2. YOLO 모델 구조
```text
입력 이미지 ─ 이미지 전처리
    ↓
Backbone ─ Edge · Texture · Shape 등의 특징 추출
    ↓
Neck ─ 여러 해상도의 특징 결합
    ↓
Head ─ 클래스 · Bounding Box 예측
    ↓
Postprocess ─ Confidence 필터링 · NMS (오늘 실습 경로)
    ↓
Results ─ 최종 탐지 결과
```

| 구성 | 역할 | 특징 |
|---|---|---|
| Backbone | 이미지 특징 추출 | 깊어질수록 해상도 감소·채널 증가 |
| Neck | 여러 크기의 특징 결합 | 작은 객체·큰 객체 대응 |
| Head | 클래스·위치 예측 | Scale별 탐지 결과 생성 |
| Postprocess | 예측 결과 정리 | `conf`, `iou`, `nms=True` |
- Multi-Scale Detection
    - 객체 크기가 다르기 때문에 서로 다른 해상도의 Feature Map을 이용
    - Neck의 핵심 역할은 서로 다른 해상도의 특징을 결합하여 다양한 크기의 객체를 탐지하도록 돕는 것
### 4-3. YOLO 추론 과정
1) 입력 이미지 읽기·크기 조정·정규화
2) 모델 순전파로 객체 후보 생성
3) Confidence 기준 이하의 후보 제거
4) NMS로 중복 Bounding Box 제거
5) 원본 이미지 좌표로 복원 및 결과 저장
### 4-4. Ultralytics 환경 설치

### 4-5. YOLO 모델 객체 생성
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print(model.task)
print(model.names)
```
- model.task: 모델의 작업 종류 (detect)
- model.names: 모델이 예측할 수 있는 클래스 이름
- yolo26n.pt: 강의에서 사용하는 사전학습 가중치
#### YOLO(model, task=None, verbose=False)
- 모델 구조와 가중치를 불러와 학습·검증·추론·추적 기능을 제공하는 객체를 생성
- `best.pt`, `yolo26n.pt` 등

### 4-6. 학습 전 데이터 검증 (Data Contract)
- 모델 학습 전에 **데이터셋의 라벨·좌표·분할 상태를 확인**해야 한다

| 검사 항목 | 통과 기준 | 오류 예시 |
|---|---|---|
| 파일 대응 | 대상 객체가 있으면 라벨 필수 | 객체가 있는데 TXT 없음 |
| 좌표 범위 | 정규화 범위 및 Box 경계 확인 | `width=1.4` |
| 박스 크기 | `w > 0`, `h > 0` | 크기가 0인 Box |
| 클래스 범위 | `class_id`는 0~N-1의 정수 | 클래스 3개인데 ID 4 |
| Split 독립성 | Group·Hash 중복 없음 | 같은 Frame이 Train·Val에 존재 |
| 데이터 분포 | 클래스·박스 크기 분포 확인 | 특정 클래스 표본 부족 |
### 4-7. 학습 하이퍼파라미터
**하이퍼파라미터**(Hyperparameter)는 **학습 전에 설정**하는 값으로, **학습 성능과 실행 비용에 영향**을 줌

| 파라미터 | 의미 | 영향 |
|---|---|---|
| **epochs** | 전체 학습 반복 횟수 | 과소적합·과적합 |
| **imgsz** | 모델 입력 이미지 크기 | 작은 객체·메모리·속도 |
| **batch** | 한 번에 처리하는 이미지 수 | VRAM·Gradient 변동 |
| device | CPU·GPU 선택 | 실행 환경·속도 |
| workers | 데이터 로딩 프로세스 수 | CPU·I/O 병목 |
| seed | 난수 시드 | 초기화·증강 순서 재현성 |
### 4-8. Data Augmentation
- 이미지를 변환할 때, **Bounding Box 좌표도 동일하게** 변환
- 좌우 반전, 조명·색상 변화, 객체 크기·위치 변화, 여러 이미지를 조합, 여러 이미지 정보를 혼합
### 4-9. YOLO 학습 (Training)
### model.train()
```python
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 1. 학습 장치 설정
    device = 0 if torch.cuda.is_available() else "cpu"

    # 2. 사전학습 모델 로드
    model = YOLO("yolo26n.pt")

    # 3. 모델 학습
    results = model.train(
        data="dataset.yaml",    #학습 데이터셋 YAML 경로
        epochs=30,              #전체 학습 반복 횟수
        imgsz=640,              #입력 이미지 크기
        batch=16,               #한 번에 학습하는 이미지 수
        device=device,          #학습 장치 (0: 첫 번째 GPU, "cpu": CPU)
        workers=0,              #데이터 로딩 작업자 수
        nms=True,               #NMS 사용 여부
        project="runs/day15",   #결과 저장 상위 폴더
        name="helmet_yolo"      #실험 결과 폴더 이름
    )

    # 4. 결과 경로 확인
    print(model.trainer.save_dir)
```
#### 학습 처리 흐름
1. 데이터셋의 images, labels, dataset.yaml 확인
2. 사전학습 Checkpoint 또는 모델 YAML 선택
3. 학습 실행 및 검증 지표 기록
4. 최적·마지막 Checkpoint와 실행 설정 저장
### 4-10. 학습 결과 파일 (Artifacts)
| 결과 파일 | 역할 |
|---|---|
| `weights/best.pt` | 검증 기준 최적 Checkpoint |
| `weights/last.pt` | 마지막 Checkpoint |
| `results.csv` | Epoch별 Loss·Metric·Learning Rate |
| PR·F1 Plot | 탐지 성능 시각화 |
| Confusion Matrix | 클래스별 예측 오류 분석 |
| Label Distribution | 클래스·라벨 분포 확인 |
| Train·Val Batch 이미지 | 데이터·라벨 시각적 검수 |
| `args.yaml` | 학습 실행 인자 기록 |
### 4-11. 핵심 정리
- **One-stage Detector**는 한 번의 순전파로 객체 후보 예측하는 것
- **Backbone**는 이미지 특징 추출한다.
- **Neck**는 여러 해상도의 특징 결합한다.
- **Head**는 클래스·Bounding Box 예측한다.
- **YOLO()**는 모델 구조와 가중치 로드한다.
- **model.train()**는 모델 학습 및 검증 결과 저장한다.
- **best.pt**는 검증 기준 최적 모델이다.
- **last.pt**는 마지막 Epoch 모델이다.
- **Data Contract** 학습 전 데이터 품질과 분할 확인
- **Hyperparameter**    학습 조건 및 실행 비용 조절
- **Augmentation**  이미지와 Bounding Box를 함께 변환


## 5. 모델 검증 (Metric, Checkpoint)
### 5-1. YOLO 검증
### model.val()
```python
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 1. 평가 장치 설정
    device = 0 if torch.cuda.is_available() else "cpu"

    # 2. 최적 모델 로드
    model = YOLO("runs/day15/helmet_yolo/weights/best.pt")

    # 3. 모델 검증
    metrics = model.val(
        data="dataset.yaml",    #데이터셋 YAML 경로
        split="val",    #평가 데이터 분할 (val, test)
        imgsz=640,      #모델 입력 크기
        device=device,  #평가 장치
        workers=0,  
        nms=True,       #NMS 적용 여부
        conf=0.001,     #예측 후보의 Confidence 기준
        iou=0.70        #NMS의 IoU 임계값
    )

    # 4. 결과 출력
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
```
### 5-2. 검증 지표 해석
| 지표 | 의미 | 확인할 내용 |
|---|---|---|
| Images | 검증 이미지 수 | 검증 표본 충분 여부 |
| **Instances** | 실제 객체 수 | 클래스별 객체 분포 |
| Precision (P) | 탐지한 것 중 맞은 비율 | FP |
| Recall (R) | 실제 객체 중 찾은 비율 | FN |
| mAP50 | IoU 0.50 기준 AP 평균 | 기본 탐지 성능 |
| **mAP50-95** | 다양한 IoU 기준 AP 평균 | 위치 정밀도 |
| **Speed** | 처리 시간 | 실시간 처리 가능성 |
### 5-3. 학습 결과를 이용한 오류 진단
| 관찰 결과 | 가능한 원인 | 우선 확인 |
|---|---|---|
| Train 개선, Val 정체 | 과적합·분포 차이 | Split·Augmentation·Epoch |
| P 높음, R 낮음 | 탐지 누락·높은 Conf | 작은 객체·라벨 누락 |
| R 높음, P 낮음 | 거짓 탐지 많음 | Negative Image·Conf |
| mAP50 ≫ mAP50-95 | Box 위치 부정확 | 라벨 밀착·해상도 |
| 클래스별 성능 편차 큼 | 불균형·라벨 규칙 차이 | 표본 수·혼동 클래스 |
### 5-4. Checkpoint — best.pt·last.pt·resume
| 구분 | 의미 | 주요 용도 |
|---|---|---|
| `best.pt` | 검증 기준 최적 Epoch의 모델 | 추론·최종 평가 |
| `last.pt` | 마지막으로 저장한 모델 | 마지막 학습 상태 확인 |
| resume | 중단된 학습 이어서 실행 | 학습 재개 |
### 5-5. 핵심 정리
- **model.val()**   모델 검증 및 성능 측정
- **Precision·Recall**  FP·FN 관점의 탐지 성능
- **mAP50·mAP50-95**    객체탐지 및 위치 정확도 평가
- **best.pt · last.pt**    최적·마지막 모델 저장 및 학습 재개


## 6. 모델 추론 (Results 객체, 오류 분석)
### 6-1. YOLO 추론
**추론**은 **학습된 YOLO 모델**을 이용해 **새로운 이미지나 영상**에서 객체의 종류와 위치를 예측하는 과정
### model.predict()
```python
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("best.pt")

# 2. 객체탐지 실행
results = model.predict(
    source="samples/",  #이미지·영상·폴더·URL·카메라 등 입력
    conf=0.25,          #Confidence 임계값 (6.6참고)
    iou=0.70,           #NMS IoU 임계값 (6.6참고)
    nms=True,           #NMS 사용 여부
    imgsz=640,          #모델 입력 이미지 크기
    save=True,          #시각화 결과 저장 여부
    stream=False        #결과를 순차적으로 반환할지 여부
)

# 3. 탐지 결과 확인
for result in results:
    print(result.path, len(result.boxes))
```
### 6-2. Results 객체
| 속성·메서드 | 설명 |
|---|---|
| result.path | 입력 파일 경로 |
| result.orig_shape | 원본 이미지 크기 `(H, W)` |
| result.names | Class ID와 클래스 이름 매핑 |
| result.boxes | 탐지된 Bounding Box 집합 |
| result.plot() | 박스·Label이 그려진 이미지 배열 |
| result.save() | 시각화 이미지 저장 |
| result.to_df() | 표 형태로 결과 변환 |
| result.save_txt() | 탐지 결과 TXT 저장 |
### 6-3. Box · Confidence · Class 읽기
```python
# 1. 탐지 결과 순회
for result in results:
    boxes = result.boxes

    # 2. Tensor를 NumPy 배열로 변환
    xyxy = boxes.xyxy.cpu().numpy() #Tensor를 CPU로 이동 #Tensor를 NumPy 배열로 변환
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)   #클래스 ID를 정수형으로 변환

    # 3. 객체별 탐지 결과 출력
    for box, score, class_id in zip(xyxy, conf, cls):
        name = result.names[class_id]
        print(name, round(float(score), 3), box)
```
### 6-4. 탐지 결과 시각화 및 저장
```python
from pathlib import Path
import cv2

# 1. 저장 폴더 생성
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)

# 2. 탐지 결과 순회
for i, result in enumerate(results):
    stem = out / f"result_{i:03d}"

    # 3. 이미지 저장
    ok = cv2.imwrite(str(stem) + ".jpg", result.plot())
    if not ok:
        raise OSError(stem)

    # 4. 예측 TXT 저장
    txt = stem.with_suffix(".txt")
    txt.write_text("", encoding="utf-8")
    result.save_txt(str(txt), save_conf=True)
```
#### result.plot()
- 탐지 Box·클래스·점수를 그린 이미지를 반환합니다.
- OpenCV에서 사용할 수 있는 BGR 배열입니다.
### 6-5. 객체탐지 오류 분석
| 오류 유형 | 관찰 예시 | 우선 확인 |
|---|---|---|
| Missed Object | 작은 사람을 놓침 | 해상도·라벨·Recall |
| **False Positive** | 나뭇가지를 사람으로 탐지 | Negative Image·Conf |
| Wrong Class | 고양이를 개로 분류 | 클래스 경계·불균형 |
| Loose Box | 박스에 배경이 많이 포함 | 라벨 품질·imgsz |
| **Duplicate Box** | 객체 하나에 여러 박스 | NMS IoU·Conf |
| Domain Shift | 야간·비·새 카메라에서 성능 악화 | 데이터 분포 |
### 6-6. Confidence Threshold, NMS IoU Threshold (하이퍼파라미터)
- **Confidence Threshold** : 탐지 결과를 유지하기 위한 **최소 예측 점수(임계점)**
- **NMS IoU Threshold** : **서로 겹치는 예측 Box를 중복으로 판단하는 기준**

Confidence는 점수를 기준으로, NMS IoU는 박스의 겹침을 기준으로 후보를 제거
### 6-7. 핵심 정리
- **model.predict()**   학습된 모델로 객체탐지 실행
- Results 값    입력별 탐지 결과를 관리하는 객체
- boxes.xyxy    원본 Pixel 좌표
- boxes.conf    예측 Confidence
- boxes.cls 예측 Class ID
- stream=True   결과를 순차 반환하여 메모리 사용 관리
- **Confidence Threshold**  낮은 점수의 후보 제거
- **NMS IoU Threshold** 중복 Bounding Box 제거
- Error Analysis    오류 유형별 원인 분석

**전체 흐름**: 모델 로드 → model.predict() → Results 확인 → Box·Class·Confidence 추출 → 결과 저장 → 오류 분석


## 7. 객체 추적 (Object Tracking, Track ID, 속도 측정)
### 7-1. 객체 추적 (Object Tracking)
**추적 Tracking** : 연속된 영상 프레임에서 **같은 객체를 연결**하고, **동일한 Track ID를 유지**하는 기술
#### Detection과 Tracking의 차이
| 구분 | Detection | Tracking |
|---|---|---|
| 입력 | 이미지 또는 개별 Frame | 연속된 Frame |
| 출력 | Box·Class·Confidence | Box·Class·Confidence·Track ID |
| 상태 | 프레임 간 상태 없음 | 이전 프레임 상태 사용 |
| 주요 오류 | Miss·FP·Wrong Class | ID Switch·Track Loss |
| 활용 | 객체 위치 확인 | 카운팅·궤적·체류시간 |
#### 추적 처리 흐름
```text
Detection ─ Frame에서 객체의 Box·Class 탐지
    ↓
Association ─ 이전 Frame과 현재 Frame의 객체 연결
    ↓
Track ID ─ 동일 객체에 식별자 유지
    ↓
Results ─ 궤적·카운팅·체류시간 분석
```
### 7-2. YOLO 영상 객체 추적 (Tracking)
### model.track()
```python
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("best.pt")

# 2. 영상 객체 추적
results = model.track(
    source="traffic.mp4",       #영상·카메라·연속 프레임 입력
    persist=True,               #호출 사이 Tracker 상태 유지 여부
    tracker="botsort.yaml",     #추적 알고리즘 설정 YAML
    conf=0.25,                  #Confidence 임계값
    iou=0.70,                   #NMS IoU 임계값
    nms=True,                   #NMS 사용 여부
    stream=True,                #결과를 순차적으로 반환할지 여부
    save=True                   #추적 결과 시각화 저장
)

# 3. Track ID 확인
for result in results:
    print(result.boxes.id)
```
### 7-3. persist와 Tracker 상태 관리
**persist**는 이전 프레임의 추적 상태를 **다음 호출에서도 유지할지 결정하는 설정**
- 주의 사항
    - Track ID는 모델 가중치에 영구 저장되지 않습니다.
    - Tracker는 추론 중 상태를 관리합니다.
    - 서로 다른 영상의 Tracker 상태를 섞으면 안 됩니다.
    - **Tracker 설정은 YAML 파일로 관리**합니다.
### 7-4. Track ID별 중심점 궤적 수집
객체의 이동 경로를 분석하려면, 프레임마다 **Track ID와 Bounding Box 중심 좌표**를 기록
```python
# 1. 모델 및 저장 구조 준비
model = YOLO("best.pt")
tracks = {}

# 2. 영상 추적
for frame_idx, result in enumerate(
    model.track(
        source="traffic.mp4",
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        nms=True
    ),
    start=1
):
    # 3. Track ID와 Box 확인
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        continue
    ids = boxes.id.int().cpu().tolist()
    xywh = boxes.xywh.cpu().tolist()

    # 4. 객체별 중심점 저장
    for tid, (cx, cy, w, h) in zip(ids, xywh):
        tracks.setdefault(tid, []).append(
            (frame_idx, cx, cy)
        )
```
### 7-5. Tracking 오류 분석
객체 추적에서는 Detection 오류 외에도 ID 연결 과정에서 오류가 발생
| 오류 | 현상 | 가능한 원인 |
|---|---|---|
| **ID Switch** | 같은 객체의 ID가 변경됨 | 가림·유사 객체·낮은 FPS |
| Track Fragmentation | 하나의 궤적이 여러 조각으로 분리 | Detection Miss·짧은 Buffer |
| False Track | 배경에 잘못된 ID 생성 | FP·낮은 Conf |
| Lost Track | 가림 후 객체를 다시 연결하지 못함 | Association 설정 |
| Duplicate Track | 한 객체에 여러 ID 생성 | 중복 Detection·NMS |
### 7-6. 속도·지연시간·처리량 측정
객체 추적을 실시간으로 실행하려면 정확도와 함께 처리 성능도 확인해야 한다.
| 지표 | 의미 | 기록 조건 |
|---|---|---|
| Latency | 한 Frame 처리에 걸린 시간 | 전처리·추론·후처리 범위 |
| FPS | 1초당 처리 가능한 Frame 수 | Batch·입력 크기·Device |
| Throughput | 단위 시간당 처리 이미지 수 | Batch Inference |
| Peak VRAM | 최대 GPU 메모리 사용량 | 모델·imgsz·batch |
| End-to-End | 영상 입력부터 결과 저장까지 전체 시간 | Decode→Track→Encode |
### 7-7. 핵심 정리
| 개념 | 핵심 |
|---|---|
| Detection | 프레임마다 객체 탐지 |
| Tracking | 프레임 사이의 동일 객체 연결 |
| Association | 이전·현재 프레임의 탐지 결과 연결 |
| Track ID | 객체별 연속 식별자 |
| **model.track()** | YOLO 객체 추적 실행 |
| **persist=True** | 호출 사이 Tracker 상태 유지 |
| BoT-SORT·ByteTrack | 객체 추적 알고리즘 |
| ID Switch | 동일 객체의 ID 변경 오류 |
| Latency·FPS | 처리 지연시간과 초당 프레임 수 |
| End-to-End | 영상 입출력까지 포함한 전체 처리 성능 |

**전체 흐름**: 영상 입력 → 객체 Detection → 이전 Frame과 Association → Track ID 부여 → 객체별 궤적 저장 → 추적 오류·FPS 분석
