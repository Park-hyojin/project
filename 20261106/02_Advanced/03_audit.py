"""
실습 03: YOLO 라벨 통계 및 데이터 분할 검증

목적: 객체 크기·비율·클래스 분포와 데이터 중복 여부 확인

[핵심 개념]
- area         : 정규화된 Bounding Box 면적 (w × h)
- aspect_ratio : 실제 픽셀 기준 가로세로비 (너비 / 높이)
- split        : Train / Validation / Test 데이터 구분
- group_id     : 같은 그룹의 데이터가 서로 다른 Split에 섞이는지 확인
- sha256       : 파일 내용의 해시값으로 정확한 중복 검사

[실행 흐름]
1. 데이터 로드 | Manifest 및 YOLO TXT 라벨 읽기
2. 라벨 분석   | 객체 면적·가로세로비 계산
3. 분포 확인   | Split별 클래스 수 집계
4. 중복 검사   | 그룹 중복 및 동일 파일 검사
5. 결과 저장   | label_audit.csv 저장

────────────────────────────────────────
■ 출력 결과 예시

class_id   0   1   2
split
test      10  10  10
train     30  30  30
val       10  10  10

split 간 그룹 중복: 0
정확히 같은 파일: 0

■ 결과 해석
- class_id 0~2 : 객체 클래스별 라벨 수
- train        : 클래스별 30개
- val / test   : 클래스별 10개
- 그룹 중복 0  : 동일 그룹이 여러 Split에 걸쳐 있지 않음
- 파일 중복 0  : 동일 SHA-256 값을 가진 파일이 없음

핵심: 클래스 분포가 균형적이며, 그룹 및 정확한 파일 중복이 발견되지 않음
"""
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
manifest=pd.read_csv(BASE/"data/manifest.csv")
rows=[]

for image in manifest.itertuples():
    for line in (BASE/"data"/image.label).read_text(encoding="utf-8").splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        
        # 요구: 정규화 면적 w*h와 원본 pixel 기준 가로세로비를 기록하세요.
        # TODO A03: 위 요구사항의 핵심 코드를 작성하세요.
        rows.append({
            "split": image.split,
            "class_id": int(cls),
            "area": w * h,
            "aspect_ratio": (w * image.width) / (h * image.height)
        })
        
r=pd.DataFrame(rows);r.to_csv(out/"label_audit.csv",index=False)
print(r.groupby(["split","class_id"]).size().unstack(fill_value=0).to_string())
print("split 간 그룹 중복:", int((manifest.groupby("group_id").split.nunique()>1).sum()))
print("정확히 같은 파일:", int(manifest.sha256.duplicated().sum()))
print("빈 라벨은 정상 negative 이미지이며 누락 라벨과 다릅니다.")
