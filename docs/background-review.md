# 개발용 약지도 배경 계약

`background_review.build`는 개발 웨이퍼 1·2·9의 제공 좌표를 수정 증거에 따라 대응시킨 후, 각 웨이퍼에서 주석 수가 가장 많은 원본 하나를 선택한다. 좌표에서 128px 이상 떨어진 128×128 후보를 생성한다. 미주석 영역을 곧바로 배경 정답으로 해석하지 않는다.

`finalize`에는 모든 후보에 대한 id/decision/note가 필요하다. `background_candidate`만 약지도 학습 후보이며, uncertain/artifact_candidate는 제외한다. 모든 결과는 Codex_AI의 형태 관찰이고 human_verified/expert_ground_truth는 false이다. 원본·제공 라벨을 수정하지 않는다. 패치 해시와 상대 경로를 검증한다.

계약은 `xrt_weak_background_v1`이다. candidates의 wafer, source_id, source_sha256, 좌표, patch sha256와 상위 cohort_manifest_sha256/coordinate_repair_sha256를 소비자가 검증해야 한다. 웨이퍼 단위 분리 및 최종 시험 웨이퍼 8 제외가 필수다. 원본 하나씩의 작은 표본이므로 전체 배경이나 실제 결함 부재를 보증하지 않는다.

BPD의 256px 문맥과 선 영역은 자동 제안만 저장한다. training_eligible=false이며 길이·각도·전환점 정답 또는 물리 단위 측정으로 사용하지 않는다. 기존 데이터셋 계약은 바꾸지 않고 별도 결과에 저장한다.
