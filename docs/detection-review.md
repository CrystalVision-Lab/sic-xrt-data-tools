# 실제 검출 후보 AI 관찰 계약

ML 품질 실험에서 높은 종류 점수·제공 좌표와 40px 초과 거리로 선택한 후보를 검토한다. 이 선택은 오검출 정답이나 무작위 표본을 뜻하지 않는다. DataTools #29에서 준비하며 소비자는 [ML #37](https://github.com/CrystalVision-Lab/sic-xrt-ml/issues/37)이다.

`detection_review.build(queue_path, sources, output)`는 원본 또는 ZIP 멤버의 SHA256와 좌표에 연결된 원본 해시를 대조한다. 128px 원본 패치와 256px 주변 문맥, 20개씩의 보기용 이미지를 저장한다. 노란 괄호는 학습 패치 범위이며 중심을 가리지 않는다. 가장자리 문맥의 검은 영역은 실제 픽셀이 아니고 context_valid_xyxy/context_clipped로 기록한다. 학습용 128px는 패딩하거나 위치를 바꾸지 않는다. 웨이퍼 8은 거부한다.

`finalize`에는 모든 ID마다 decision/note가 필요하다. weak_background만 약지도 이진 판별기의 배경 후보로 허용한다. possible_defect는 타입 확정이 아니며 uncertain과 함께 학습에서 제외한다. 관찰은 Codex_AI, human_verified=false, expert_ground_truth=false다. 원본이나 제공 라벨을 바꾸지 않는다. 그림/파일 내용이 바뀌면 해시 검증에서 거부한다.

출력 계약은 `xrt_detection_observations_v1`이다. rows 대신 candidates 배열을 사용하며 id/wafer/path/sha256/source_sha256/좌표/decision/training_eligible를 포함한다. 기존 xrt_weak_background_v1 무작위 배경과 별도 계약이다. ML이 둘을 합칠 때 웨이퍼 단위로 분리하고 출처별 성능을 따로 보고해야 한다. 같은 물체 주변의 여러 후보가 있을 수 있어 패치 수를 독립 물체 수로 해석하지 않는다. 배경 후보가 한 웨이퍼에만 있으면 이물에 대한 교차 웨이퍼 검증이 부족하다.
