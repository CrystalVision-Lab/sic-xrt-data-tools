# 전체 주석 복원과 검수 작업 계약 v2

`python -m sic_xrt_data_tools.annotation_workspace --registry <source_registry v1> --output <새 폴더>`

등록 기록 파일의 해시와 사용한 원본/ROI 내용 해시를 대조한 후, 원본 ROI의 큰 종류 및
세부 문자를 복원합니다. 원본 내부, 원본 등록 내부, 기존 출력 폴더에는 생성하지 않습니다.
원본과 기존 데이터셋/학습 분할은 변경하지 않습니다. 입력 source_registry가 incomplete이면 거절합니다.

같은 ROI 바이트라도 영상 문맥이나 라벨 파일명이 다르면 서로 다른 해석을 보존합니다.
같은 영상·큰/세부 종류·좌표가 같은 점은 한 점에 참조를 모으고, 같은 좌표의 다른 종류는
불일치로 기록합니다. 원본 좌표는 변형하지 않고 EXIF 반전도 적용하지 않습니다.
영상 연결은 같은 단계의 동일 폴더 단일 TIFF를 우선하고, ROI의 명시적 맨 앞 영상 번호와
JPEG 이름이 하나로 대응되는 경우 또는 단일 영상인 경우를 후보로 선택합니다.
폴더·이름은 물리적 동일 시야를 증명하지 않으므로 모든 연결은 시각 검수 전 후보입니다.
원본이 바뀌면 실패하고, 변한 내용을 ROI 파싱 오류로 숨기지 않습니다.

## 산출물

- workspace.json: 계약 v2, 원본 등록 해시, 종류/Area별 분포, 검수자 표시 이름, 실제 생성 주체.
- images.jsonl: 모든 Area의 원본 영상과 3D 스택, 바이트 복사본 참조.
- provider_points.jsonl / csv: 좌표·큰 종류·세부 문자·원본 ROI 참조·검수 상태.
- roi_mappings.json: 영상 연결 근거, 대안 영상, 원본 ROI 좌표 집합 해시.
- annotation_exclusions.json / label_conflicts.json: 빈/연결 보류 주석과 좌표/종류 불일치.
- review/ 및 전체웨이퍼_주석검수.html: 원본 방향의 개요와 종류별 최대 8점 예시.
- review_template.csv: 사람이 실제 확인한 결과를 별도로 입력할 수 있는 템플릿.
- requirements_status.json: 검출/경로/전후/3D/스케일 및 평가 정답의 결손 상태.
- output_hashes.json: 모든 기록 및 검수 이미지의 SHA-256 (자기 자신 제외).

검수자 표시 이름과 실제 작업 주체는 다릅니다. 생성 주체는 Codex_AI이고, 검수 템플릿의
actual_actor는 비워 둡니다. 제공자의 이름 라벨을 복원해도 human_verified는 false,
eligible_for_verified_evaluation은 false입니다. 원본 제공자의 완전 주석 범위나 물리적
a~f 정의, 좌표 대응 의미까지 자동으로 확인했다고 주장하지 않습니다.
unknown 점과 주석 없는 Area/영역을 정상으로 바꾸지 않습니다.

검수 이미지는 원본 픽셀에서 잘라 만든 보기 자료입니다. uint8은 색/밝기 그대로이며,
그 밖의 dtype은 화면 표시용 대비 변환만 적용합니다. 패치가 경계에 걸리면 실제 영상 범위만
표시하고 좌표 표시 위치도 함께 변환합니다. 모든 점이 아니라 종류/영상별 일부 예시를 보여 줍니다.

큰 종류와 세부 종류의 학습 계약은 각각 별도입니다. a~f를 이미지 각도 구간으로 새로
만들거나 회전/반전된 패치에 원래 문자를 붙이지 않습니다. split은 모두 unassigned이고,
training_ready는 false입니다. 소비자는 검수·교정·분할 상태가 맞는 데이터를 선택해야 합니다.
