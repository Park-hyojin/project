
"""
실습 05: YOLO26 모델 검증 (model.val)

목적: model.val()을 이용한 객체 탐지 모델 평가 및 성능 지표 이해

[핵심 개념]
- model.val() : Validation 데이터로 모델 성능 평가
- Precision   : 탐지한 객체 중 올바른 탐지 비율
- Recall      : 실제 객체 중 올바르게 탐지한 비율
- AP          : 클래스별 Precision-Recall 곡선 아래 면적
- mAP50       : IoU 0.5 기준 클래스별 AP 평균
- mAP50-95    : IoU 0.5~0.95 기준 AP 평균

[실행 흐름]
1. 데이터 설정 | Validation 경로·클래스 정의 → YAML 저장
2. 모델 로드   | demo.pt 사전 준비 가중치 불러오기
3. 모델 검증   | model.val()로 Validation 데이터 평가
4. 지표 추출   | 전체 mAP 및 클래스별 Precision·Recall·AP 계산
5. 결과 저장   | JSON·CSV 파일로 평가 결과 저장

────────────────────────────────────────
■ model.val() 주요 파라미터
- data  : 데이터셋 YAML 파일 경로
- split : 평가할 데이터 분할 (val / test / train)

■ 평가 결과
- metrics.box.map50        : 전체 mAP50
- metrics.box.map          : 전체 mAP50-95
- metrics.box.class_result(): 클래스별 Precision·Recall·AP50·AP50-95
- metrics.box.ap_class_index: 평가 결과에 포함된 클래스 ID

핵심: model.val()로 검증 데이터를 평가하고 전체·클래스별 탐지 성능을 확인하는 방법
"""
from pathlib import Path
import json
import yaml
import torch
import pandas as pd
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":
    torch.set_num_threads(1)
    out = BASE / "outputs"
    out.mkdir(exist_ok=True)

    # 1. 데이터셋 설정 및 YAML 파일 생성
    config = {
        "train": str(BASE / "data/images/train"),
        "val": str(BASE / "data/images/val"),
        "names": ["square", "circle", "triangle"]
    }
    path = out / "dataset_active.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # 2. 사전 준비된 YOLO 모델 로드
    model = YOLO(str(BASE / "data/demo.pt"))

    # 3. Validation 데이터로 모델 성능 평가
    metrics = model.val(
        data=str(path),
        split="val"
    )

    # 4. 전체 mAP50 및 mAP50-95 추출
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "split": "val"
    }
    (out / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 5. 클래스별 Precision·Recall·AP 추출
    rows = []
    for i, cls in enumerate(metrics.box.ap_class_index):
        pr, rc, ap50, ap = metrics.box.class_result(i)
        rows.append([
            int(cls), model.names[int(cls)],
            float(pr), float(rc), float(ap50), float(ap)
        ])

    # 6. 클래스별 평가 결과 CSV 저장
    pd.DataFrame(rows, columns=["class_id", "name", "precision", "recall", "ap50", "ap50_95"]).to_csv(out / "class_metrics.csv", index=False)

    print(summary)