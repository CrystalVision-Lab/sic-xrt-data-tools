# 점 검수 이력 계약 v1

화면 소비자: Analyzer [Issue 31](https://github.com/CrystalVision-Lab/sic-xrt-analyzer/issues/31).
검증 소유자: Data Tools [Issue 9](https://github.com/CrystalVision-Lab/sic-xrt-data-tools/issues/9).
서로의 내부 소스를 import하거나 복사하지 않고 다음 파일 계약으로 연결한다.

입력: annotation_workspace v2와 candidate_workbench v1. 둘의 output_hashes.json 및 후보가 기록한
workspace_manifest_sha256이 일치해야 한다. 원본과 기존 작업은 불변이다.

검수 세션은 독립 폴더에 review_session.json과 decisions.jsonl을 저장한다. 세션 헤더:
schema=`review_decisions_session`, schema_version=1, workspace_manifest_sha256,
candidate_manifest_sha256, reviewer_display_name. 표시 이름은 실제 확인한 사람의 서명이 아니다.

각 JSONL 이벤트 필수 필드:

- event_id: 고유 문자열, sequence: 1부터 연속 정수, previous_event_sha256: 첫 이벤트는 null.
- item_id: 원본 point_id 또는 candidate_id. image_asset_id, source_kind(provider/contrast),
  frame_index(2D null), x,y는 해당 입력 항목과 일치한다. 좌표를 이동시키는 계약이 아니다.
- decision: confirm/correct/exclude/hold. confirm은 제공자 원래 종류와 같을 때만 가능하다.
- reviewed_label: BPD, TED/TSD, TED_a~f, TSD_a~c, normal/dust/scratch 또는 보류용 null/unknown.
  확인·수정에는 unknown이 아닌 종류가 필요하다. 기본 TED/TSD를 임의의 세부 문자로 바꾸지 않는다.
- actual_actor: 실제 확인 주체, actor_type: human/ai, directly_checked: boolean,
  reviewed_at: 시간대가 포함된 ISO 시각, notes: 선택 메모.
- event_sha256: 이 필드 자신을 뺀 JSON 객체를 ensure_ascii=false, sort_keys=true,
  separators=(',', ':'), allow_nan=false로 UTF-8 직렬화하여 SHA-256 계산한 값.

화면은 이미지를 읽지 못하면 직접 확인 체크를 비활성화한다. 확인/수정은 체크 및 실제 사람 이름이
필요하고 후보 종류를 기본 선택하지 않는다. 실제 입력은 append-only로 flush/fsync하여 기록하고,
같은 항목의 이후 결정은 앞의 기록을 덮어쓰지 않는다. 다른 창에서 파일이 바뀌면 재시작을 요구한다.
해시는 이력 변경 검사용이며 암호학적 서명 또는 검수자의 전문성을 인증하는 수단이 아니다.

`python -m sic_xrt_data_tools.review_decisions --workspace <v2> --candidates <v1> --session <검수 폴더> --output <새 폴더>`

검증기는 이벤트 해시·순서·중복 ID·항목 참조·좌표·종류·확인 주체·확인 여부·시각을 검사한다.
검수 이력이 없어도 남은 수와 결손을 포함한 보고서를 생성한다. 검수 도중 새 이벤트가 추가되어도
검증한 스냅샷의 로그 해시를 기록하며 새 기록을 검사했다고 주장하지 않는다.
readiness.json, reviewed_points.jsonl, input_hashes.json, output_hashes.json을 생성한다.

사람이 직접 확인·수정한 최신 점만 point_human_reviewed=true다. 이후 보류/제외 또는 AI 결정이면
이전의 사람 승인으로 계속 쓰지 않는다. AI 이름을 human으로 제출한 기록은 거부한다.
점 검수를 영상 전체 완전 주석, 전문가 종류 기준 확인, 물리적 동일 결함, 물리 보정의 승인으로
승격하지 않는다. 모든 점의 split은 unassigned, eligible_for_verified_evaluation은 false다.
training_ready 및 full_dataset_export_complete는 현재 항상 false다. 원본 부족 조건이 해소되지
않은 상태에서 이 파일을 최종 학습·평가 정답으로 자동 소비하면 안 된다.

원본 픽셀 패치 표시 및 사용자 조작은 Analyzer가 소유한다. 데이터 검증 결과는 모델 학습 실행이나
배포 승인을 대신하지 않는다. 세부 기준과 물리 메타데이터는 현재 제공된 자료만으로 확정되지 않았다.
