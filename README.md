# sic-xrt-data-tools

대용량 XRT/TIFF 데이터의 검사, 타일링, 메타데이터·스케일 추출, 주석 변환, 데이터셋 검증, Before/After 비교 보조 도구를 담당합니다.

## 시작

- Python 3.11 이상
- 개발 도구: `python -m pip install -e ".[dev]"`
- 이미지 처리: `python -m pip install -e ".[image]"`
- 검사: `ruff check .`, `pytest`

현재는 `tiff/`, `tiles/`, `metadata/`, `annotations/`, `validation/`, `comparison/` 모듈 경계만 마련했습니다. 각 변환의 실제 CLI와 라이브러리 API는 Issue에서 입력·출력 규약을 정한 뒤 추가합니다.

원본 데이터와 생성 데이터는 Git에 넣지 않습니다. ML 또는 Analyzer 저장소의 코드를 이 저장소 작업 중 수정하지 않습니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.
