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

## 전체 원본 등록

`python -m sic_xrt_data_tools.source_registry --source <원본 폴더> --output <새 결과 폴더>`로
원본·ZIP·중첩 ROI 파일의 해시/CRC와 영상 전체 페이지 읽기를 검사합니다.
실제 바이트 중복과 이름 기반 전후/확대 후보를 분리하여
로컬 JSON·CSV·HTML 기록을 만듭니다. [원본 등록 계약 v1](docs/source-registry-v1.md)을 따르세요.
파일 읽기 성공과 결함 정답 검수를 구분하며 기존 라벨·분할·학습을 변경하지 않습니다.

## 제공자 세부 주석 복원

`python -m sic_xrt_data_tools.annotation_workspace --registry <원본 등록 폴더> --output <새 폴더>`로
TED a~f·TSD a~c·BPD 점, JPEG 연결 후보와 검수 자료를 복원합니다.
[주석 작업 계약 v2](docs/annotation-workspace-v2.md)에 따라 출처·검수·좌표 상태를 분리합니다.

## 전체 웨이퍼 검수 후보

복원된 주석 작업에서 라벨이 없는 영상까지 대비·국소 선·전후 정렬·3D 경로 후보를 생성합니다.
자동 결과는 정답으로 승격하지 않고, 모든 후보의 실제 검수 입력표와 웨이퍼 단위 평가 계획을
함께 저장합니다. [후보 작업 계약 v1](docs/candidate-workbench-v1.md)을 참고하세요.

## 점 검수 이력 검증

Analyzer의 검수 세션을 새 폴더의 점 검수 상태 보고서로 변환합니다.
[검수 계약 v1](docs/review-decisions-v1.md)에 따라 이력·참조·좌표·확인 주체를 검증하며
점 확인을 영상 전체 또는 물리적 정답의 확정으로 자동 승격하지 않습니다.

원본 데이터와 생성 데이터는 Git에 넣지 않습니다. ML 또는 Analyzer 저장소의 코드를 이 저장소 작업 중 수정하지 않습니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.
