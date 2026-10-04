# 원본 반전과 ROI 좌표 보정 계약

같은 파일명의 JPG와 TIFF라도 좌표계가 같다고 가정할 수 없다. 폴더 안 TIFF가 하나라는 기존 연결 근거는 후보 선택일 뿐, 주석 정합성 증명이 아니다.

`coordinate_repair.exact_horizontal_mirror`는 전체 디코딩 픽셀을 64행 블록으로 비교한다. 정확한 좌우 반전 일치일 때만 증거를 반환한다. 이 증거만으로 모든 ROI를 반전시키면 안 된다. 각 ROI 파일이 어느 이미지 좌표계를 사용하는지 별도로 확인하고, 확인된 point ID만 명시적으로 지정한다. 변환은 0부터 시작하는 픽셀 중심 좌표에서 `new_x = width - 1 - old_x`, `new_y = old_y`다.

## 로컬 원본에서 확인한 결과

- 웨이퍼 2 before의 같은 이름 JPG/TIFF는 전체 12197×12148×3 디코딩 픽셀이 좌우 반전 후 정확히 같다.
- TSD a/b/c의 총 853개 제공자 POINT ROI는 JPG 방향에 대응한다. TED a~f와 BPD는 TIFF 방향에 대응한다. 동일 이미지에 연결된 ROI 사이에서도 좌표 방향이 섞여 있었다.
- TSD a의 상하 공간 분할 평균 영상 모두 JPG에서는 중심에 강한 신호가 있고, 기존 TIFF 좌표에서는 일관된 중심 신호가 없었다. 다른 같은 폴더 JPG 3개도 대조했다. 반전 후 TIFF의 신호와 원본 JPG의 결함이 대응한다.
- 모델 예측 또는 검증 점수는 좌표 변환을 찾는 데 사용하지 않았다. 타입 라벨은 변경하지 않았다. 세부 타입의 물리학적 정답을 새로 확정한 것은 아니다.

## 버전과 보존

`repair_dataset`은 기존 schema-v1 데이터셋을 새 디렉터리에 복사하고 선택한 패치만 다시 추출한다. 원본 데이터셋의 전체 파일 해시를 전후 검증한다. 원래 표본, 타입 라벨, train/val/test 분할은 보존한다. 시험셋 보정은 이 경로에서 거절한다. 기존 잘못된 패치를 보고 남긴 AI 검수는 새 패치에 승계하지 않는다.

추가 계약 `coordinate_repair.json` schema_version=1:

- `evidence`: 원본 ID/파일 해시/전체 픽셀 반전 증거와 개별 ROI 판정 근거
- `all_provider_corrections`: point ID별 old_xy/new_xy/ROI 원본 참조/제공자 세부 타입
- `selected_patch_changes`: 기존 선택 표본의 패치별 old/new CSV 행
- `original_type_labels_changed=false`, `cohort_changed=false`, `reserved_test_modified=false`
- `research_only=true`, `human_verified=false`, `test_evaluated=false`

기존 dataset schema는 1을 유지한다. manifest/patch 해시는 새 버전에 맞게 갱신한다. 보정 전 품질 기준으로 뽑힌 표본이라는 선택 편향이 남아 있다. `export_corrected_point_audit`는 별도 audit.json 계약으로 전체 보정 주석을 내보내며, 이미지 경계 밖 패치만 제외한다. 예측 점수로 표본을 제외하지 않는다. 이는 기존 시험셋을 대체하지 않는다.

## 논문 근거의 범위

Harada et al., *Non-destructive identification of edge-component burgers vector of threading dislocations in SiC wafers by birefringence imaging*, Diamond and Related Materials 138 (2023) 110192, [DOI](https://doi.org/10.1016/j.diamond.2023.110192).

이번 작업에서는 출판사 공개 초록·방법/결과/결론 발췌를 재확인했다. 공개 원고 PDF 다운로드는 완료되지 않았으므로 전체 도표를 다시 읽었다고 주장하지 않는다. 논문은 복굴절 영상과 XRT의 결합, 촬영 조건을 이용한 Burgers vector 판별을 다룬다. 그 근거를 색·밝기만의 자동 재라벨 규칙으로 일반화하지 않았다. 이번 수정의 결정적 근거는 논문에서 추측한 타입이 아니라 사용자 원본의 정확한 반전 관계다.
