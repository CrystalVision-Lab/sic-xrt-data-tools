# 전체 웨이퍼 후보 작업 계약 v1

`python -m sic_xrt_data_tools.candidate_workbench --workspace <annotation_workspace v2> --output <새 폴더>`

설치: `pip install -e ".[workbench]"`. 입력은 등록된 원본의 바이트별 대표 영상과 복원된
제공자 점 주석이다. 원본 등록과 주석 작업 산출물의 SHA-256을 확인하고, 사용한 실제 원본도
해시 및 읽기 전후 파일 상태를 검증한다. 원본·기존 패치·분할·주석 작업 폴더는 수정하지 않는다.
출력은 새 독립 폴더로만 생성한다. 모든 산출물 해시는 output_hashes.json에 기록한다.

## 후보 생성 범위

- 단일 2D 영상마다 밝은/어두운 대비 극값을 각각 제안한다. 기본 상한은 합계 64점이다.
  긴 변을 최대 4096 정도로 원본 정수 간격 샘플링하고, Gaussian sigma 1/6의 차이와
  MAD×4 또는 응답 상위 0.5% 기준 및 국소 극값을 사용한다. 공간 셀별 상한으로 집중을 줄인다.
  출력 좌표는 원본 좌표이고 detector_stride로 샘플링 간격을 기록한다.
  이미 제공자 점이 있는 영상은 기존 점 반경 16 원본 픽셀 안 후보를 제외한다.
- 제공자 BPD 점 부근 반경 96 원본 픽셀에서 대비 성분의 주성분 축을 구한다.
  긴 축/짧은 축 분산비가 4 이상인 국소 직선 성분만 제안한다. 색 대비가 강하거나 성분이
  원형이거나 점 근처에 없으면 보류한다. 선의 전체 경로·분기·곡률은 확정하지 않는다.
- 동일 Area의 before/after x100 개요 영상이 각각 하나인 경우, 512×512 보기 영상의
  정수 이동을 위상 상관으로 제안한다. 실제 같은 시야, 배율과 EXIF 방향은 확인 전이다.
  이동 후 영상 상관이 0.3 이상일 때만 같은 밝기 부호의 가까운 후보를 전후 대응으로 제안한다.
  전후 대응 거리는 보기 좌표 단위이며, 변환 방향은 after에서 before로 이동하는 방향이다.
- 3D TIFF의 모든 프레임을 개별 처리한다. 긴 변 약 1536, 기본 상한 프레임당 24점이다.
  연속 프레임 간 같은 대비 부호의 상호 최근접점(24 원본 픽셀 이하)을 연결한다.
  최소 3개 연속 프레임의 연결만 경로 후보로 기록한다. 프레임은 물리적 깊이로 가정하지 않는다.

알고리즘은 검수용 제안이다. 대비가 센 먼지·스크래치·주석 표시를 잘못 잡거나 희미한 결함을
놓칠 수 있다. bright/dark는 TED/TSD/BPD 분류가 아니고, contrast_score는 결함 확률이 아니다.
빈 후보 목록과 주석 없는 영역은 정상/결함 없음 정답이 아니다. 상한 때문에 후보 수를 전체
결함 개수·결함 밀도나 검출 재현율로 해석할 수 없다. JPG/TIFF·배율·프레임 사이의 물리적
동일 결함은 바이트 중복 제거만으로 확정되지 않으므로 서로 다른 촬영본 후보를 합산하지 않는다.

## 산출물 및 불변 조건

- candidate_summary.json: 실행 범위, 알고리즘 설정, 검수 상태와 산출물 개수.
- candidates_2d.jsonl / candidates_3d.jsonl: 원본 영상/프레임 참조와 원본 좌표, 대비 부호.
- bpd_local_line_candidates.jsonl: 제공자 BPD 점별 국소 선 후보 또는 보류 이유.
- survey_coverage.jsonl: 처리된 영상/프레임, 후보 수, 미확정 주석 범위. 손상 파일은 실패 기록.
- before_after_alignment_candidates.json / before_after_identity_candidates.json: 정렬 및 대응 후보.
- stack_neighbor_candidates.json / stack_path_candidates.json: 프레임 연결 및 경로 후보.
- candidate_review_template.csv: 전 후보 검수 입력표. 검수자 표시 이름은 작업 계약에서 가져오되
  실제 확인 주체, 검수 시각, 종류/세부 문자와 주석은 비워 둔다.
- future_wafer_folds.json: 9개 웨이퍼 각각을 한 번씩 남기는 계획. 실제 분할을 바꾸지 않는다.
- reviews.json / review/ / 전체웨이퍼_후보검수.html: 영상 개요와 일부 좌표 예시.

base_label/subtype/probability는 null, human_verified 및 eligible_for_verified_evaluation은 false,
split은 unassigned, training_ready는 false이다. 자동 경로에 µm 길이나 conversion_depth_um을
넣지 않는다. 각도는 원본 +X 기준의 0~180°이며 step-flow 0°나 a~f 정의와 같다고 가정하지 않는다.
표시 예시는 각 영상/대비 부호별 최대 8점으로, 모든 후보를 시각 검증한 기록이 아니다.

전체 주석 범위·종류 정의·촬영본 대응을 검수한 후에만 정답 데이터로 내보낼 수 있다. 점 주석의
사람 확인만으로 영상 전체를 완전 주석 또는 물체 검출/분할 평가 정답으로 승격하지 않는다.
미확인 3D↔Area 대응은 평가 분할에 포함하지 않는다. 학습 설정 선택은 바깥 평가용 웨이퍼를
보지 않고 나머지 웨이퍼 내부의 그룹 검증으로 진행해야 한다. 모든 바깥 점수를 반복 확인하며
설정을 고른 경우에는 그 점수를 독립 최종 평가라고 보고할 수 없다.

SciPy 공식 API: [Gaussian 필터](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html),
[국소 최대 필터](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html).
이 문서는 해당 API로 구현한 후보 생성 규칙을 설명한다. 결함 종류 판별의 물리적 근거를 대신하지 않는다.
