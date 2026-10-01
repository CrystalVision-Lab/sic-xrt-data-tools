# 점 주석 패치 계약 v1

## 목적

단일 페이지 2D TIFF에서 ImageJ 점 주석 중심의 분류 패치 **후보**를 만듭니다.
클래스는 ROI 파일명의 BPD/TED/TSD에서 가져옵니다. 클래스별 사각형/마스크나
정상 주석이 아닙니다. 원본과 점 주석의 시각적 정합은 pending으로 남깁니다.

```powershell
python -m pip install -e ".[dataset,dev]"
python -m sic_xrt_data_tools.point_dataset --source ORIGINAL_ROOT --output NEW_VERSION_FOLDER --splits splits.json
python -m sic_xrt_data_tools.validate_point_dataset NEW_VERSION_FOLDER
python -m sic_xrt_data_tools.balanced_subset NEW_VERSION_FOLDER
```

입력 폴더 아래 `2D XRT/숫자 Area.zip`과 선택적인 `3D XRT/*.tif`를 읽습니다.
splits.json은 `{"1":"train","2":"val","8":"test","9":"train"}` 형태이며
웨이퍼가 서로 다르다는 근거가 있어야 합니다. 원본 경로 내부 출력과 기존 버전
폴더의 덮어쓰기를 거절합니다. 실패 시 BUILD_FAILED.json이 남고 학습하지 않습니다.

## 좌표·매칭

- ImageJ decoder로 서브픽셀 좌표를 보존하고 point ROI만 받습니다.
- 같은 폴더의 유일한 TIFF 후보를 연결합니다. 여러 TIFF가 있을 때 before/after
  이름으로 좁히며, 반대 단계·모호한 후보·페이지가 다른 ROI는 보류합니다.
- 매칭 근거는 폴더/단계이며 물리적 정합을 보증하지 않습니다. 미리보기 검수가 필요합니다.
- 중첩 RoiSet.zip을 읽고 원본/클래스/소수점 4자리 좌표별 중복을 제거합니다.
- 클래스가 다른 점이 같은 패치 안에 있으면 해당 패치를 보류합니다. 표시되지 않은
  결함은 탐지할 수 없으므로 이 규칙만으로 완전한 단일 클래스 영상을 보장하지 않습니다.
- 패치 중심은 floor(x+0.5), floor(y+0.5)입니다. 원본 ROI 좌표와 crop left/top을 기록합니다.
- 경계 밖 패치는 padding하지 않고 제외합니다. 원본 픽셀/dtype을 보존하고 TIFF로 저장합니다.
- 전체 스택은 사용하지 않으며 이미지 하나씩 임시 파일에 풀고 읽습니다. 가능한 경우
  읽기 전용 memmap을 사용합니다. 큰 압축 페이지의 경우 한 페이지 전체 디코딩 RAM이 필요합니다.
- ZIP 경로를 파일 경로로 사용하지 않고 고정된 임시 TIFF 이름으로만 추출합니다.

## 출력

`patches_128/분할/클래스/*.tif`, `patches_256/...` 두 크기는 같은 점에서 파생된
서로 다른 실험 입력입니다. 서로 독립적인 표본으로 중복 계수하지 않습니다.
두 크기의 같은 원본/웨이퍼는 항상 같은 분할입니다. 검수 화면에만 표시를 덧그립니다.

`기록/samples.csv` 필드:

| 필드 | 의미 |
|---|---|
| patch_id / point_id / source_id | 재현 가능한 ID |
| path / sha256 | 루트 상대 TIFF 경로와 파일 SHA-256 |
| wafer / split / label / size | 원본 그룹·분할·ROI 클래스·패치 크기 |
| x / y / left / top | 원본 좌표와 패치 좌상단 |
| dtype | 원본을 보존한 dtype |
| review_status / label_basis | pending / ImageJ_point_ROI_filename |

`points.csv`는 모든 매칭 점과 경계/중복 참조를 기록하고 `sources.json`은 원본
TIFF 해시·경로·크기·dtype·축·분할을, `archives.json`은 입력 ZIP 해시를 기록합니다.
CSV는 Windows 호환 UTF-8 BOM, JSON은 UTF-8입니다. 각 파일은 개인정보/실제 자료를
포함할 수 있으므로 Git 제외 외부 출력 폴더에만 저장합니다.

`summary.json`의 schema_version=1은 이 출력 계약 버전입니다. ML 소비자는 버전·
파일 해시·클래스 목록·분할 및 pending 검수 상태를 확인해야 합니다. 모형 입력 전처리와
학습 코드는 ML 저장소의 별도 기능 범위이며 직접 import/소스 복사는 하지 않습니다.

`검수.html`은 원본 축소 영상에 주석을 겹친 화면과 원본/중심 표시 패치 샘플을 제공합니다.
색이 덧그려진 원본이나 정합이 틀린 영상은 학습에서 제외해야 합니다.
`color_pixel_fraction_sampled`는 색 픽셀의 간이 보고값이며 주석 혼입의 확정 판정이 아닙니다.
제외 주석, 중복 점, 미사용 2D 이미지와 무주석 3D 자료 목록을 별도 JSON에 저장합니다.

클래스 수가 크게 다를 때 `balanced_subset`은 train에서만 각 클래스의 최소 개수만큼
seed=42로 선택하는 CSV 뷰를 생성합니다. 원래 패치를 삭제/복사하지 않고 검증·시험
분포도 변경하지 않습니다. train_balanced_128.csv와 train_balanced_256.csv를 이용하거나
전체 train을 클래스 가중치와 함께 쓰는 방식은 후속 ML 구현에서 선택합니다.

## 검증

생성 TIFF의 해시/shape/dtype과 samples.csv 수, 원본별 단일 분할을 검사합니다.
테스트에서는 실제 원본 불변, 좌표·패치 픽셀 동일성, 혼합 클래스/경계 제외,
중첩 중복, 모호한 매칭과 원본 내부 출력 거절을 검증합니다. 실제 클래스 의미·
누락 주석·독립 웨이퍼 관계는 자동 검사만으로 확정하지 않습니다.
