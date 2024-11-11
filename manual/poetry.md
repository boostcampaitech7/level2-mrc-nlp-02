# poetry manual

  ** poetry란? ** 

프로젝트 관리를 간편하게 도와주는 도구로서 의존성 관리에 강점이 있음.

<b>장점</b> : 명확한 버전 및 관계 명시. poetry.lock 파일을 통해 버전끼리의 충돌 방지
간단한 cli 사용법. 의존성 그룹화 지원(ex. --dev 옵션)

nodejs npm과 거의 유사

<b>단점</b> : PyPI 인덱스 문제가 간혹 생길 수 있음. 새로 배포된 버전을 다운받을려고 할 때 갱신이 안 되어서 다운 못 받는 현상이 생길 수 있음. (poetry version 문제일 가능성 큼. poetry self update로 해결 혹은 pip install로 다운 후 poetry로 import)


## Installation

1. 가상 환경 생성 
```sh
python -m venv py_venv
```
2. 가상 환경 활성화
3. poetry 설치
```sh
pip install poetry
```
4. poetry 패키지 설치 - 이미 pyproject.toml이 존재할 경우 pyproject.toml 안에 설정된 패키지들을 설치
```sh
poetry install
```
5. (옵션) poetry 프로젝트 초기화 - * 이미 pyproject.toml 파일이 있으면 실행할 필요 없음
```sh
poetry init
poetry lock
``` 
6. (옵션) poetry로 requirements.txt import하여 의존성 패키지에 추가 및 설치
```sh
poetry add $(cat requirements.txt)
```




## Manual

1. poetry로 특정 라이브러리 설치
```sh
poetry add {package}
```
2. poetry로 최신 패키지를 다운 받지 못 할 때 혹은 poetry를 업데이트할 때
```sh
poetry self update
```
3. pyproject.toml과 poetry.lock 파일이 존재하면 이 파일들에 명시된 의존성 설치
```sh
poetry install
```
4. 현재 설치된 패키지들을 최신 버전으로 업데이트하고 poetry.lock 파일 갱신할 때(이 명령은 프로젝트원과의 합의 및 검수 필요)
```sh
poetry update
```
5. 의존성 제거할 때 (패키지 설치 제거 및 패키지 관리 파일 업데이트)
```sh
poetry remove {package}
```
6. 현재 설치된 패키지들의 목록과 정보 확인
```sh
poetry show 
poetry show {package}
```

7. 현재 프로젝트를 빌드하여 배포 가능한 패키지 생성
```sh
poetry build
```

8. poetry 패키지 캐시 삭제(의존성 설치 중 문제 발생하거나 캐시가 잘못된 경우 수행)
```sh
poetry cache clear [--all pypi]
```

9. 가상환경 정보 확인 - 현재 프로젝트의 가상환경 정보 등을 확인
```sh
poetry env info
```

10.  프로젝트 유효성 검사 - pyproject.toml의 설정이 올바른지, 의존성 트리가 유효한 지 검사
```sh
poetry check
```