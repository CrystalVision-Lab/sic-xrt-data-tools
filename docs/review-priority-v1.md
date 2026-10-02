# 첫 검수 계획 v1

제공자 라벨은 그대로 유지한다. 영상·세부 종류별 최대 3점의 공간 분산 진단 표본과
알려진 좌표 오류·종류 불명·같은 좌표의 종류 충돌을 우선 대상으로 고른다.
이 계획은 확률 표본이나 정확도 추정이 아니며 표본 확인으로 전체 항목을 승인하지 않는다.
자동 대비 후보는 첫 검수에서 미뤄 두고 삭제하거나 정답으로 바꾸지 않는다.

`python -m sic_xrt_data_tools.review_priority --workspace <v2> --candidates <v1> --output <새 폴더>`

출력 review_plan.json은 schema=review_priority_plan, schema_version=1,
workspace_manifest_sha256, candidate_manifest_sha256, items(item_id/reasons), selected_count,
total_items, per_group, selection, audit 및 제한사항을 포함한다. output_hashes.json으로 묶는다.
Analyzer는 두 입력 해시·ID 목록·중복·개수를 확인하고 선택 목록으로만 첫 화면을 제한한다.
기존 decisions.jsonl, 원본, 라벨, 학습/평가 분할은 바꾸지 않는다. 선택하지 않은 항목과
주석 없는 웨이퍼는 여전히 미확인이다. AI/사람 확인이나 학습 준비를 생성하지 않는다.

관련 화면: https://github.com/CrystalVision-Lab/sic-xrt-analyzer/issues/35
