# sic-xrt-data-tools

대용량 XRT/TIFF 데이터의 검사, 타일링, 메타데이터·스케일 추출, 주석 변환, 데이터셋 검증, Before/After 비교 보조 도구를 담당합니다.

## 시작

- Python 3.11 이상
- 개발 도구: `python -m pip install -e ".[dev]"`
- 이미지 처리: `python -m pip install -e ".[image]"`
- 검사: `ruff check .`, `pytest`

`tiff/`, `tiles/`, `metadata/`, `annotations/`, `validation/`, `comparison/` 모듈 경계를 둡니다. 각 변환의 CLI와 라이브러리 API는 Issue에서 입력·출력 규약을 정한 뒤 추가합니다.

원본 데이터와 생성 데이터는 Git에 넣지 않습니다. ML 또는 Analyzer 저장소의 코드를 이 저장소 작업 중 수정하지 않습니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.

## 제공자 세부 종류 패치 데이터셋

`python -m sic_xrt_data_tools.provider_subtype_dataset build --source "<원본>\2D XRT" --output <Git 밖 새 폴더>`로
ImageJ 점 ROI의 BPD / TED a~f / TSD a~c 라벨을 보존한 128px uint8 RGB 패치, Area 단위 분할,
manifest와 미리보기를 만듭니다. `validate <폴더>`로 해시·형태·분할을 검사합니다.
라벨은 제공자 라벨이며 전문가 확정이 아닙니다. [계약 v1](docs/provider-subtype-dataset-v1.md)을 따르세요.
