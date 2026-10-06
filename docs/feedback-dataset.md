# 프로그램 검수 → 학습 자료

Analyzer Issue #54의 공개 출력 계약 `xrt_feedback_v1`을 받는다. ML Issue #43에서 사용할 `xrt_feedback_dataset_v1`을 만든다.
역할은 자료 검증·추출이며 모델 학습이나 UI 처리는 하지 않는다. 기존 자료와 원본은 변경하지 않는다.

```powershell
python -m sic_xrt_data_tools.feedback_dataset --reviews "검수/feedback.json" "다른영상/feedback.json" --output "새자료폴더"
```

여러 세션 파일을 받고 같은 원본 해시·페이지의 내보내기는 같은 세션의 동일 이력 접두부인 경우 최신본만 사용한다.
세션/웨이퍼/수정 이력이 충돌하면 거부한다. 최초 주석이나 전문가 정답을 조용히 덮어쓰지 않는다.
원본 파일 SHA-256·크기·페이지·RGB dtype와 모든 이벤트의 revision/previous_target/target를 검증한다.
XRT JPG는 EXIF 반전 없이 파일 좌표를 유지하고 TIFF는 지정한 페이지를 추출한다. TIFF↔JPG의 반전 대응을 추측하지 않는다.

출력 dataset.json은 샘플/기록 해시, 원본·검수 파일 해시 및 계약을 담는다. samples.json은 학습 마스크와 패치 해시를 담는다.
records.json과 reviews/에 원래 예측·수정 이력이 남는다. excluded.json에 제외 이유가 남는다.
폴더가 이미 있으면 거부하고 새 폴더를 요구한다. 추출 실패로 dataset.json이 없는 폴더는 미완료이며 학습 입력으로 사용할 수 없다.

| 확인 상태 | 유무 학습 | 종류 학습 | 위치 학습 |
|---|---|---|---|
| 결함 있음, 종류·위치 미확정 | 포함 | 제외 | 제외 |
| 결함 있음, 종류 확인 | 포함 | 포함 | 위치 확인 시 포함 |
| 결함 아님 | 포함 | 제외 | 제외 |
| 보류·중복·되돌려 미확정 | 제외 | 제외 | 제외 |

`objectness_mask`, `type_mask`, `location_mask`는 독립이다. 예측 종류를 정답으로 사용하지 않고 표시되지 않은 곳을 배경으로 만들지 않는다.
확인된 결함 중심 32px 이내의 배경 기록은 충돌 우려로 제외한다. 이는 자동 결함 판단이 아니라 보수적인 학습 제외다.
현재 계약은 128×128 RGB uint8, 원본 픽셀 x/y, EXIF 회전 없음, 페이지 0 기반이다.
위치 수정은 수정 좌표 패치와 원래 좌표 패치를 모두 추출한다. 보정값은 **목표 좌표 − 실제 추출 중심(left+64, top+64)**이다.
원래 패치를 같이 쓰므로 보정 학습이 전부 0이 되는 오류를 피한다. 마스크가 꺼진 위치/종류는 손실 계산에 쓰지 않는다.
추출 중심에서 축별 32px를 넘는 이동과 잘린 가장자리 패치는 제외 이유를 남긴다.
좌표만으로 BPD 선 윤곽·bounding box·물리 길이를 만들지 않는다.

웨이퍼 8은 입력 검증 단계에서 금지한다. 다른 웨이퍼도 번호와 원본 대응을 실제로 확인해야 한다.
`expert_ground_truth=false`, `exhaustive_annotations=false`, `test_evaluated=false`를 유지한다.
사용자가 입력한 종류는 human_review 근거이며 전문가 확인을 보장하지 않는다.
