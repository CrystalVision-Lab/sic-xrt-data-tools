# AI 검수 반영 연구 후보 계약

관련 Issue: #17. 소비자 학습 작업은 [ML #11](https://github.com/CrystalVision-Lab/sic-xrt-ml/issues/11)에서 수행한다.

`reviewed_research_dataset`은 annotation_workspace v2와 provider_point_quality_audit v1,
시간순으로 명시한 AI 검수 배치를 읽어 기존 ML 소비자와 호환되는 schema-v1 중심 패치 후보를 만든다.
자동 품질 경고, 가장 최근 AI 보류, 제공자 기본 타입과 AI 제안의 불일치,
128픽셀 패치 안에 다른 타입 주석이 함께 있는 항목을 제외한다.
이웃 주석 검사는 제외된 주석까지 포함한 전체 제공자 좌표를 사용한다.

AI 재판독은 이전 관측을 지우지 않고 history에 함께 보존하며 최신 관측을 선택에 적용한다.
기존 제공자 라벨과 좌표는 수정하지 않는다. AI 제안이 다른 경우 자동 정답 수정 대신 제외한다.
검수되지 않은 제공자 ROI도 품질 기준을 통과하면 연구 후보로 포함한다.
자동 검사 통과나 AI와 제공자 일치를 전문가 확정 정답으로 간주하지 않는다.

원본 해시를 확인한 SourceReader에서 128×128 패치를 원래 uint8/uint16 형식으로 저장한다.
십자가, 표시용 확대, 회색조 스트레치는 학습 픽셀에 들어가지 않는다.
새 독립 출력만 허용하며 실패한 결과에는 BUILD_FAILED.json을 기록한다.
각 패치 해시·형태·자료형·웨이퍼/주석 분리 검증을 통과하고 모든 split에 세 클래스가 있어야 완료한다.

| 계약 파일 | 의미 |
|---|---|
| summary.json | schema_version=1, task=center_point_patch_classification_candidates, 클래스 순서 BPD/TED/TSD, 개수와 한계 |
| 기록/samples.csv | 패치 경로/해시, 제공자 좌표와 기본/세부 타입, phase, AI 관측 유무, 실제 actor |
| 기록/sources.json | source_id, wafer, split, shape, dtype, sha256, locator, phase |
| 기록/excluded_points.jsonl | 모든 제외 주석과 이유 목록; 이유별 개수는 중복 가능 |
| 기록/ai_review_history.json | 재판독을 포함한 전체 AI 관측; 실제 actor Codex_AI, human/expert=false |
| provenance.json | 입력 manifest 해시, 시간순 검수 해시, samples.csv 해시, 픽셀/좌표 정책 |
| validation.json | 구조 검증 결과; 의미상 전문가 검증을 뜻하지 않음 |
| balanced_subsets.json 및 기록/train_balanced_128.csv | seed42로 학습만 클래스별 최소 개수에 맞춘 선택 뷰 |
| output_hashes.json | 생성 파일 해시; 입력과 기존 파일은 보존 |

웨이퍼 1·9=train, 2=val, 8=test를 고정한다. 전후 영상도 같은 웨이퍼 그룹에 남는다.
검증/시험을 균형 샘플링하지 않는다. 웨이퍼 3~7은 타입 주석이 없어 포함하지 않는다.
a~f/a~c 물리 정의, 전후 동일 결함 대응, 물리 스케일과 3D 깊이 정답은 확보되지 않았다.
이 데이터셋은 이들 과제의 정답 데이터셋이나 전체 웨이퍼 검수 완료 결과가 아니다.

2026-10-03 실행: 제공자 17,147점, AI가 직접 본 서로 다른 점 386개,
재판독을 포함한 관측 535개. 최신 관측은 제안 197/보류 189개이며 전문가 확정은 0개다.
총 8,778패치(학습 7,123/검증 360/시험 1,295)를 생성했다.
학습 BPD146/TED3,103/TSD3,874, 검증 BPD33/TED257/TSD70,
시험 BPD6/TED780/TSD509다. 균형 학습 뷰는 각146개, 총438개다.
포함된 패치 중 직접 AI 판독은153개이며 나머지8,625개는 자동 검사 통과 제공자 라벨 후보다.
시험 BPD가6개라 향후 시험 지표의 불확실성이 크다. 평가 결과를 확정 정확도로 해석하지 않는다.

실행 예시(경로는 환경에 맞춰 지정):

```powershell
python -m sic_xrt_data_tools.reviewed_research_dataset --workspace WORKSPACE --quality AUDIT --ai-review FIRST_BATCH --ai-review NEXT_BATCH --output NEW_OUTPUT
```

검수 패널 prepare의 `--context-size 512`는 넓은 원본과 무주석 확대,
표시용 회색조 스트레치를 제공한다. 기본값256은 기존 패널과 동일하다.
재판독 원본 좌표는 보존하고 스트레치만으로 타입을 확정하지 않는다.
