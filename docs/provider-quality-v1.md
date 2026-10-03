# 전체 제공 주석 품질 검사 v1

입력은 annotation_workspace v2와 ai_visual_type_proposals v1 출력 디렉터리입니다.
입력 파일 해시, AI 패널에 고정된 작업 해시, 주석 수·ID·원본 타입·좌표 대응을 검사합니다.
실제 원본은 SourceReader가 바이트 해시를 검증한 후 읽습니다. 출력은 원본과 모든 입력의
안팎에 겹치지 않는 새 디렉터리로 제한하며 원본 주석·사람 검수·모델을 쓰지 않습니다.

provider_point_quality_audit v1은 quality_summary.json, point_quality.jsonl/CSV,
examples.json, 전체주석_품질검사.html, 예시 패치와 SHA-256 출력 해시를 제공합니다.
입력 작업 및 기존 AI 결과의 manifest SHA를 고정하며 입력 계약을 변경하지 않습니다.
원본 주석과 타입을 보존하고 automatic_inspected, inspection_origin=automatic_image_metrics,
flags, priority_score와 이미지 판독 여부를 별도로 기록합니다. 기존 AI 결과에 들어 있지
않은 항목은 ai_visual_reviewed=false이며 자동 검사만으로 판독 완료를 만들지 않습니다.
human_verified/expert_semantic_confirmation/eligible_for_verified_evaluation/training_ready는
false입니다. 원래 AI 타입 제안은 출처 연결 목적으로만 가져옵니다.

좌표를 이동하지 않고 중심 주변 최대 128픽셀 원본을 읽습니다. uint8/uint16만 허용하며
정규화는 각각 255/65535를 사용합니다. 다른 형식은 실패하고 완료 보고서를 생성하지
않습니다. 영상 밖 좌표는 coordinate_invalid, 경계에서 잘린 문맥은 context_clipped입니다.

청록색/보라색 픽셀 비율 10% 이상이고 어느 한 행의 비율이 50% 이상이면
color_band_suspected입니다. 이 색 정책은 대표 패널에서 본 띠를 찾아내기 위한 경고이며
색 자체가 오류라는 의미는 아닙니다. 국소 Laplacian 에너지 0.00008 미만은
low_local_detail입니다. 촬영 조건과 배경이 평탄한 경우에도 발생할 수 있어 흐림을
확정하거나 자료를 자동 삭제하지 않습니다.

좌표 주변 최대 64픽셀의 Gaussian(σ=1)-Gaussian(σ=4) 절대 대비를 계산합니다.
대비 기준은 max(0.015, 4×MAD×1.4826), 중심 반경은 12픽셀입니다. 중심 원 안에 기준을
넘는 대비가 없으면 center_contrast_not_clear, 가장 가까운 국소 극값이 반경 밖이면
contrast_offset_suspected입니다. 극값은 물리 결함 수·동일 개체·정답 타입이 아니며
별도의 결함으로 좌표를 자동 이동하지 않습니다. 작은 TED, 어두운 선, 주변 큰 대비를
놓치거나 잡음을 검출할 수 있습니다. 대비 부족과 이탈 지표의 중복 경고는 가능합니다.

같은 이미지에서 소수 4자리 좌표가 같은 주석과 서로 다른 문자 라벨은 중복/충돌
경고로 기록합니다. 근처의 서로 다른 점은 이 검사로 중복으로 합치지 않습니다.
priority_score는 경고 가중치의 합으로서 정확도나 타입 예측 확률이 아닙니다.

예시는 영상별 각 경고 유형과 공간적으로 떨어진 경고 없는 위치를 포함하며 통계적
확률 표본이 아닙니다. 경고 없음은 검증된 타입, 학습 가능 확정, 전체 웨이퍼 주석 완료를
뜻하지 않습니다. 3D·미주석 영상·전후 동일 결함 대응은 이 검사 범위 밖입니다.
