"""
실습 04: YOLO26 API 학습 및 CLI 명령 비교

목적: model.train()과 CLI를 이용한 YOLO 학습 설정 및 실행 방법 이해

[핵심 개념]
- Python API : Python 코드에서 model.train()으로 학습 실행
- CLI        : 터미널에서 yolo detect train 명령으로 학습 실행
- dataset.yaml : Train·Validation 경로 및 클래스 정보 정의
- Smoke Test : 짧은 학습으로 정상 실행 여부 확인

[실행 흐름]
1. 데이터 설정 | 이미지 경로·클래스 정의 → YAML 저장
2. 모델 생성   | YOLO("yolo26n.yaml")로 모델 초기화
3. API 학습    | model.train()으로 2 Epoch 학습
4. CLI 명령    | 동일 학습 옵션의 CLI 명령 생성 및 저장
5. 결과 확인   | 학습 가중치 저장 경로 출력

────────────────────────────────────────
■ 주요 학습 옵션
- optimizer : 가중치 갱신 알고리즘 (SGD)
- lr0       : 초기 학습률
- seed      : 난수 시드 설정
- amp       : 혼합 정밀도 학습 사용 여부
- pretrained: 사전학습 가중치 사용 여부
- plots     : 학습 결과 그래프 생성 여부

■ API / CLI 비교
API : model.train(data=..., epochs=2, imgsz=160)
CLI : yolo detect train data=... epochs=2 imgsz=160

핵심: 동일한 학습 옵션을 Python API와 CLI 두 방식으로 설정하는 방법
"""
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":
    torch.set_num_threads(1)
    out = BASE / "outputs"
    out.mkdir(exist_ok=True)

    # 1. 데이터셋 설정 및 YAML 파일 생성
    config = {"train": str(BASE / "data/images/train"),
              "val": str(BASE / "data/images/val"),
              "names": ["square", "circle", "triangle"]}
    path = out / "dataset_active.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # 2. YOLO26n 모델 생성 (사전학습 가중치 미사용)
    model = YOLO("yolo26n.pt")

    # 3. API를 이용한 모델 학습 (n Epoch Smoke Test)
    model.train(
        data=str(path),
        epochs=20,
        imgsz=320,
        pretrained=True,
        batch=8,
        workers=0,
        device="cpu",
        amp=False,
        optimizer="SGD",
        lr0=0.01,
        seed=42,
        plots=False,
        project=str(out),
        name="smoke",
        exist_ok=True
    )

    # 4. 동일 학습 설정의 CLI 명령 생성 및 저장 (실행하지 않음)
    command = f'yolo detect train model=yolo26n.yaml data="{path}" epochs=2 imgsz=160 batch=4 workers=0 device=cpu amp=False pretrained=False optimizer=SGD lr0=0.01 seed=42 plots=False project="{out}" name=cli_smoke exist_ok=True'
    (out / "same_settings_cli.txt").write_text(command, encoding="utf-8")

    # 5. 학습 가중치 저장 경로 확인
    print("best/last:", out / "smoke/weights", "CLI 비교는 선택 실행입니다.")
