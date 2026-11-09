"""
실습 02: Confidence Threshold 및 NMS 설정 비교

개요: Confidence와 NMS IoU 임계값에 따른 객체 탐지 결과 비교
목적: 고전 NMS의 중복 제거 원리와 Precision·Recall 변화 이해

[핵심 개념]
- Confidence Threshold : 신뢰도가 낮은 예측 Box 제거
- NMS                  : 같은 클래스의 중복 Box 제거
- NMS IoU Threshold    : 두 Box의 IoU가 임계값을 초과하면 낮은 점수의 Box 제거
- TP / FP / FN         : 정답 매칭 결과를 이용한 탐지 성능 평가
- Precision / Recall   : 예측 정확도와 실제 객체 탐지 비율 평가

[실행 흐름]
1. 데이터 준비   | 예측·정답 Box 로드 → Tensor 변환
2. 신뢰도 필터링 | Confidence Threshold 이상인 Box 선택
3. NMS 적용      | 클래스별 중복 제거 → 원본 인덱스 복원
4. 정답 매칭     | 클래스 일치 + IoU ≥ 0.5 + 1:1 매칭
5. 성능 평가     | TP·FP·FN 집계 → Precision·Recall 계산
6. 결과 저장     | 설정별 비교 결과를 nms_comparison.csv로 저장

주의: 고전 NMS를 별도로 실험하며, YOLO26 기본 추론에 NMS가 적용된다고 가정하지 않음
"""
from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou, batched_nms
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
p=pd.read_csv(BASE/"data/boxes_fixture.csv");g=pd.read_csv(BASE/"data/boxes_gt.csv")
boxes=torch.tensor(p[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
scores=torch.tensor(p.confidence.to_numpy(),dtype=torch.float32)
classes=torch.tensor(p.class_id.to_numpy(),dtype=torch.int64)
gb=torch.tensor(g[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
rows=[]
for conf,nms_iou in [(0.25,0.5),(0.5,0.5),(0.25,0.95)]:
    ids=torch.where(scores>=conf)[0]
    
    # 요구: 같은 클래스끼리만 중복을 제거하는 batched_nms를 적용하세요.
    # 반환 index는 ids로 줄인 배열 기준이므로 다시 원본 index로 바꿉니다.
    # TODO A02: 위 요구사항의 핵심 코드를 작성하세요.
    keep_idx = batched_nms(boxes[ids], scores[ids], classes[ids], nms_iou)
    keep = ids[keep_idx]
    
    used=set();tp=0
    for i in keep.tolist():
        ovs=box_iou(boxes[[i]],gb)[0]
        eligible=[j for j in range(len(g)) if int(classes[i])==int(g.iloc[j].class_id) and j not in used]
        for j in sorted(eligible,key=lambda j:-float(ovs[j])):
            if ovs[j]>=0.5:used.add(j);tp+=1;break
    n=len(keep);fp=n-tp;fn=len(g)-tp
    rows.append(dict(conf=conf,nms_iou=nms_iou,kept=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0,recall=tp/len(g)))
r=pd.DataFrame(rows);r.to_csv(out/"nms_comparison.csv",index=False);print(r.round(4).to_string(index=False))
