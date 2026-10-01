# sic-xrt-data-tools

대용량 XRT/TIFF 데이터의 검사, 타일링, 메타데이터·스케일 추출, 주석 변환, 데이터셋 검증, Before/After 비교 보조 도구를 담당합니다.

## 시작

- Python 3.11 이상
- 개발 도구: `python -m pip install -e ".[dev]"`
- 이미지 처리: `python -m pip install -e ".[image]"`
- 검사: `ruff check .`, `pytest`

`tiff/`, `tiles/`, `metadata/`, `annotations/`, `validation/`, `comparison/` 모듈 경계를 둡니다. 점 주석 패치 변환과 검증은 아래 CLI로 실행하고, 그 밖의 변환은 기능 Issue에서 입력·출력 규약을 정한 뒤 추가합니다.

## 점 주석 분류 데이터셋

ImageJ point ROI와 단일 페이지 TIFF에서 원본 dtype을 유지하는 128/256 패치 후보,
웨이퍼별 분할, 추적 기록, 중복/보류 보고서와 검수 페이지를 생성합니다.
[계약 v1과 실행 방법](docs/point-dataset-v1.md)을 따르세요. 기존 출력 덮어쓰기와
원본 내부 생성을 거절하고, 주석 없는 곳을 정상으로 만들지 않습니다.

원본 데이터와 생성 데이터는 Git에 넣지 않습니다. ML 또는 Analyzer 저장소의 코드를 이 저장소 작업 중 수정하지 않습니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.
