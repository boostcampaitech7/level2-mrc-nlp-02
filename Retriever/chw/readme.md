

## Sparse test 방법

1. 프로젝트 루트 디렉터리로 이동
  ```sh
  cd 프로젝트 디렉터리
  ```
2. Retriever/chw/script 파일의 run_sparse.sh의 명령줄 인수 설정
  ```sh
    export PYTHONPATH=$PYTHONPATH:.
    python Retriever/chw/sparse_test.py --dataset_name ./data/train_dataset --data_path ./data --context_path wikipedia_documents.json --model_name_or_path klue/bert-base --method bm25 --topk 10
  ```
  
  ★ 기본적으로 프로젝트 루트디렉터리 기준으로 데이터를 인식하고 임베딩 파일을 생성하기 때문에 루트 디렉터리에서 파일을 실행하는 것이 중요

  임베딩 메서드는 --method 인수 설정 - [bm25, tkidf]를 지원<br>
  모델 및 토크아니저는 --model_name_or_path 인수 설정<br>
  topk는 --topk 인수 설정 <br>

  현재 테스트 타깃을 train 데이터와 valid 데이터 중 따로 선택하는 기능은 없기때문에 파일에서 직접 설정하셔야 합니다. 

3. ./Retriever/chw/script/run_sparse.sh를 명령줄에 입력
4. 수행 후 data 폴더에 bm25, tkidf 임베딩 생성 후 test까지 연속 실행
  