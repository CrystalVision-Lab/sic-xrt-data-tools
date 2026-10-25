# 제공자 세부 종류 점 패치 데이터셋 계약 v1

관련 Issue: #35. 스키마 이름 `provider_subtype_point_dataset`, `schema_version = 1`.

제공자가 ImageJ 점 ROI 파일명으로 남긴 큰 종류(BPD/TED/TSD)와 세부 종류
(TED a~f, TSD a~c)를 보존한 중심점 분류 패치 데이터셋입니다. 라벨은 **제공자 라벨**이며
전문가 확정 정답이 아닙니다 (`labels_source: provider`, `expert_verified: false`,
`research_only: true`). 주석이 없는 곳을 정상(배경)으로 만들지 않습니다.

```powershell
python -m pip install -e ".[dataset,dev]"
python -m sic_xrt_data_tools.provider_subtype_dataset build --source "<원본>\2D XRT" --output C:\XRT-datasets\provider-subtype-v1
python -m sic_xrt_data_tools.provider_subtype_dataset validate C:\XRT-datasets\provider-subtype-v1
```

선택 인자: `--splits splits.json` (`{"<Area 번호>": "train|val|test"}`),
`--orientation-overrides overrides.json` (`{"<ROI 상대 경로>": "identity|hflip"}`, Git 밖 보관),
`--preview-per-label N`. 출력 폴더가 이미 있거나, 원본 안이거나, Git 작업 트리 안이면 거절합니다.
실패하면 `BUILD_FAILED.json`이 남으며 그 폴더는 사용하지 않습니다.

## 입력 해석

- `2D XRT/<N> Area/...` 아래의 `.roi`, `.tif`, `.jpg`만 읽습니다. ZIP은 읽지 않습니다.
- 라벨은 ROI 파일명에서만 읽습니다. `BPD`, `TED(a)`~`TED(f)`, `TSD(a)`~`TSD(c)`.
  세부 문자가 없거나 범위를 벗어나면 제외합니다.
- 영상 연결: ROI와 같은 폴더(또는 `RoiSet`의 상위 폴더)의 TIFF. TIFF가 여럿이면
  ROI 이름의 Before/After 단계로 좁힙니다. ROI 이름의 영상 번호(`an_9_3`의 `3`)가
  TIFF 이름과 다르면 경고로 남깁니다.
- ROI 파일 단위 제외 이유(`manifest.json`의 `excluded_roi_files`, `roi_files.csv`):

| reason | 의미 |
|---|---|
| `empty_file` / `not_an_imagej_roi` / `not_a_point_roi` | 읽을 수 없는 ROI |
| `unknown_label`, `missing_subtype`, `invalid_subtype`, `unexpected_subtype` | 라벨 해석 불가 |
| `duplicate_bytes` | 다른 ROI와 바이트가 같음 (`duplicate_of`) |
| `duplicate_coordinates` | 바이트는 다르지만 같은 영상·라벨·좌표 (예: 이름만 다른 복사본) |
| `x100_magnified_image_different_scale` | x100 확대 타일 폴더. 배율이 달라 섞지 않음 |
| `no_tiff_in_directory*`, `ambiguous_tiff_*`, `roi_stage_conflicts_with_tiff` | 영상 연결 불가 |
| `coordinate_grid_undetermined` | 아래 좌표계 판정이 결론나지 않음 |

## 픽셀 출처

제공 TIFF는 같은 이름 JPEG를 디코딩해 **좌우 반전한 비트 동일 사본**입니다 (실측: 손상이 없는
TIFF 5개 모두 모든 행이 완전히 일치). 따라서 TIFF가 JPEG보다 더 원본에 가까운 무손실 자료는
아니며, 두 형식의 픽셀 값은 같습니다. 영상마다 다음을 검사해 `images.json`의
`pixel_source`/`pixel_check`에 기록합니다.

1. TIFF의 모든 행이 `transform(JPEG)`와 완전히 같으면 TIFF를 그대로 사용합니다
   (`tiff_bit_identical_to_jpeg_<transform>`).
2. 일부 행(1% 이상)만 같으면 TIFF 사본이 손상된 것이므로 같은 transform으로 JPEG에서
   다시 만든 영상을 사용합니다 (`tiff_damaged_rebuilt_from_jpeg`, `pixel_source=jpeg_hflip`).
   이는 손상되지 않은 TIFF가 가졌을 픽셀과 같습니다. 다른 픽셀 수, 순백(255) 픽셀 수,
   ROI 점 2px 이내 비율을 함께 기록합니다.
3. 완전히 같은 행이 없으면 TIFF를 그대로 쓰고 저해상도 상관만 기록합니다.

JPEG 디코더는 Pillow이며 버전을 기록합니다. 패치는 리사이즈·정규화·패딩 없이 원래 uint8 RGB
값입니다. 표시용 대비 변환은 미리보기에만 적용합니다.

## 좌표계

- ImageJ 서브픽셀 좌표(픽셀 모서리 원점): 픽셀 `i`는 `[i, i+1)`, 중심은 `i+0.5`.
  정수 좌표만 있는 ROI는 `+0.5`로 같은 규약에 맞춥니다.
- 모든 출력 좌표 `x, y`는 **TIFF 픽셀 격자**입니다. `roi_x, roi_y`는 제공 좌표 그대로입니다.
  `coordinate_transform`이 `identity`면 `x = roi_x`, `hflip`이면 `x = width - roi_x`.
- ROI 파일마다 좌표가 TIFF 격자인지, 반전된 원본 JPEG 격자인지 판정합니다. 점 위치 15×15
  국소 표준편차의 중앙값을 같은 점을 8방향으로 이동한 대조군의 중앙값으로 나눈 비율을
  두 격자에서 계산합니다. 점이 15개 이상이고, 한쪽 비율이 1.15 이상이며 다른 쪽보다
  1.10배 이상 크면 그 격자로 정합니다. 점이 적은 파일은 같은 영상·같은 큰 종류의 판정이
  하나로 일치할 때만 그 판정을 따릅니다 (`inherited_from_same_image_same_class_files`).
  라벨 내용이나 모델 점수는 사용하지 않습니다. 근거는 `roi_files.csv`의 `contrast_probe`에 있습니다.
- 패치 중심은 `center = floor(x), floor(y)`, 패치는 `left = center_x - 64`, `top = center_y - 64`.
  128px 안의 중심 64px은 `[32:96, 32:96]`입니다.

## 점 단위 규칙

- 같은 영상·같은 세부 라벨·소수점 4자리 같은 좌표는 하나의 점으로 합치고 참조를 모두 남깁니다.
- 같은 좌표에 서로 다른 라벨: `same_coordinate_conflicting_labels`로 제외.
- 중심 32×32 안에 다른 라벨(큰 종류 또는 세부 종류가 다른) 점: `other_label_within_center_32px`로 제외.
  128px 안의 이웃 수는 `neighbors_*` 열에 남기며 제외하지 않습니다.
- 영상 밖 점: `point_outside_image`. 128px 패치가 경계를 넘으면 `patch_crosses_image_boundary`.

## 분할

Area(웨이퍼) 단위입니다. 같은 Area의 Before/After 영상은 같은 물리 영역이므로 항상 같은
분할입니다. 기본값은 이전 data-tools 연구 데이터셋(#17, #25)과 같은 `1·9=train, 2=val, 8=test`입니다.
지도에 없는 Area는 `unassigned`. 웨이퍼가 4개뿐이므로 ML에서는 `area` 열로 웨이퍼 단위
교차 검증을 할 수 있습니다.

## 출력

| 파일 | 의미 |
|---|---|
| `manifest.json` | 스키마·버전, 코드 리비전(git commit, dirty), 원본 루트, 원본 TIFF/JPEG SHA-256, ROI 파일 SHA-256, 규칙·임계값, 분할, 개수, 제외, 한계 |
| `samples.csv` | 포함된 패치 한 줄씩 (아래 필드). 해시는 `manifest.samples_sha256` |
| `points.csv` | 모든 고유 점과 status/reason |
| `roi_files.csv` | 모든 ROI 파일, 사용/제외 이유, 중복 대상, 좌표계 판정 근거 |
| `images.json` | 영상별 크기, 해시, JPEG 쌍, 행 일치율, 픽셀 출처, 손상 진단 |
| `patches_128/<split>/<fine_label>/<patch_id>.tif` | 128×128×3 uint8 무압축 TIFF |
| `preview/contact_sheet.png` | 세부 라벨별 표본 8개 + 전체 평균(대비 확장). 가장자리 눈금이 중심 |
| `preview/overview_<image_id>.png` | 1/8 축소 영상 위 점 (포함=종류 색, 제외=노랑) |
| `output_hashes.json` | 자기 자신을 뺀 모든 출력 파일 SHA-256 |

`samples.csv`(UTF-8 BOM) 주요 필드: `patch_id`, `path`, `sha256`, `image_id`, `image`, `area`(=`wafer`),
`stage`, `split`, `label`(큰 종류), `subtype`(a~f, BPD는 빈 값), `fine_label`(`BPD`, `TED_a` …),
`x`, `y`, `roi_x`, `roi_y`, `coordinate_transform`, `pixel_source`, `center_x`, `center_y`, `left`,
`top`, `size`, `dtype`, `channels`, `neighbors_*`, `roi_files`, `roi_refs`, `labels_source`,
`expert_verified`, `review_status`.

## 호환성과 소비자 확인

- 새 독립 계약입니다. main에 없는 미병합 `point_dataset` v1(#3)과 폴더 구조가 다르고
  (`patches_128/<split>/<fine_label>/`, `기록/` 폴더 없음), 중심 규칙이 `floor(x)`입니다
  (이전 초안은 `floor(x+0.5)`, 최대 1px 차이). `label` 열의 의미(BPD/TED/TSD)는 같습니다.
- 소비자(sic-xrt-ml)는 `schema`, `schema_version`, `samples_sha256`, `output_hashes.json`을
  확인하고 `validate`가 통과한 폴더만 사용해야 합니다. 이 저장소 코드를 import하지 않고
  CSV/JSON/TIFF만 읽습니다.
- 스키마 필드를 지우거나 의미를 바꾸면 `schema_version`을 올립니다. 필드 추가는 v1 안에서 허용합니다.

## 2026-10-26 로컬 실행 결과

실제 자료는 Git 밖 로컬 폴더에만 있습니다. 경로·해시는 그 폴더의 manifest에만 기록합니다.

- 주석이 연결된 영상 8개(웨이퍼1 After 1~4·Before 1, 웨이퍼2·8·9 Before 1). 고유 점 17,056개, 패치 16,732개.
  제외 점: 경계 288개, 중심 32px 안의 다른 라벨 36개.
- TIFF 8개 중 5개는 반전 JPEG와 비트 단위로 같았고, 3개는 손상돼 JPEG에서 다시 만들었습니다.
  - 웨이퍼1 After 1: 351픽셀만 다르며 모두 순백(255)이고 ROI 점 2px 안에 있었습니다. 즉 TIFF에
    주석 표시가 새겨져 있어, 그대로 쓰면 패치 중심에 라벨 흔적이 들어갑니다.
  - 웨이퍼1 After 2: 약 91% 행이 색 띠와 행 이동으로 손상. After 3: 약 61% 행 손상.
- 웨이퍼2의 TSD a~c ROI는 반전된 원본 JPEG 격자였고(`hflip`), 나머지 ROI는 모두 TIFF 격자였습니다.
  반전을 보정한 뒤 세부 종류의 평균 패치 모양은 웨이퍼 사이에서 일관됩니다.
- 제외 ROI 파일: 빈 파일 2개(웨이퍼9 TED e 두 사본), `unknown` 1개(9점), x100 확대 BPD 1개(82점),
  바이트 중복 10개(웨이퍼8 RoiSet 9개와 웨이퍼9 TSD b 1개), 좌표 중복 7개(웨이퍼9의 `9_3` 이름 파일. 같은 좌표가
  `9_1` 이름으로도 있고, 점 좌표가 3번 영상 폭을 넘어 1번 영상에 속함).

| split (Area) | BPD | TED a | b | c | d | e | f | TSD a | b | c | 합계 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train (1, 9) | 411 | 381 | 2,046 | 3,353 | 1,662 | 68 | 25 | 2,775 | 2,797 | 10 | 13,528 |
| val (2) | 178 | 38 | 61 | 118 | 161 | 29 | 20 | 829 | 6 | 2 | 1,442 |
| test (8) | 32 | 311 | 464 | 196 | 93 | 50 | 56 | 187 | 373 | 0 | 1,762 |

## 한계

- 제공자 라벨이며 TED a~f / TSD a~c의 물리적 정의는 이 데이터셋에서 확인하지 않았습니다.
- 희소 세부 종류가 있습니다 (예: TSD c 12개, TED e·f 각 100~150개 수준). 시험 Area에는
  일부 세부 종류가 없습니다. 정확도·F1을 확정 성능으로 해석하면 안 됩니다.
- 점 주석이 없는 Area 3~7과 3D 자료는 포함하지 않습니다. 배경 클래스는 없습니다.
- 실제 자료, 패치, 미리보기, 실제 경로·해시가 담긴 manifest는 Git에 넣지 않습니다.
