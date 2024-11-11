# Open-Domain Question Answering (ODQA) 

## 프로젝트 개요

 Question Answering(QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 다양한 QA 시스템 중, Open-Domain Question Answering(ODQA)은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가됩니다.  
 본 프로젝트에서 사용될 모델은 two-stage로 구성됩니다. 첫 단계는 질문에 관련된 문서를 찾아주는 ‘Retriever’ 단계이고, 두 번째 단계는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 ‘Reader’ 단계입니다. 두 가지 단계를 각각 구성하고 적절하게 통합하여, 주어진 질문에 대하여 알맞은 문서를 찾아 정답을 반환하는 ODQA 시스템을 만드는 것이 이번 프로젝트의 목표입니다. 

## 팀명: 시작이 반2조 (NLP - 2조)
팀원: 강전휘, 권기태, 권지수, 박수빈, 이인설, 최현우

## 주요 특징

- Sparse Retriever와 Dense Retriever를 결합한 Hybrid Retriever 구현
- 다양한 사전학습 모델을 활용한 Reader 구현
- 앙상블 기법을 통한 성능 향상

## 평가 지표

- Retriever: Hit-k, MRR (Mean Reciprocal Rank)
  -	Hit
  	상위 k개의 검색 결과 내에 관련 문서가 존재하는지 확인하는 지표입니다.
  		관련 문서가 상위 k개 내에 포함된다면 1, 포함되지 않으면 0을 반환합니다.
  본 프로젝트에서는 상위 k개 문서 내에 관련 문서가 포함되어야 하기 때문에 Retriever에서 가장 중요한 지표입니다.
  -	MRR(Mean Reciprocal Rank)
  상위 k개의 검색 결과 내에 관련 문서의 순위의 역수를 측정하는 지표입니다. Hit는 순위를 고려하지 않기 때문에 순위를 고려할 수 있는 지표로써 도입하였습니다.

- Reader: EM (Exact Match), F1 Score
  -	EM(Exact Match)
  모델이 예측한 답변이 실제 정답과 정확히 일치하는지 측정하는 지표입니다. 답변과 정답이 100% 일치해야 점수를 받기 때문에 Reader는 정답이 존재하는 정확한 시작지점과 끝지점을 예측하는 것이 중요합니다.
  -	F1 Score
  예측된 답변과 실제 정답 단어 사이의 중복되는 단어를 측정하게 됩니다. 예측과 정답이 정확하게 일치하지 않더라도 중복되는 단어가 있다면 점수를 받을 수 있으므로 EM보다는 조금 더 유연한 평가지표입니다. 


## 주요 실험 결과

### Retriever

- Sparse Retriever: BM25 알고리즘이 우수한 성능을 보임
- Dense Retriever: multilingual-e5-large-instruct 모델이 좋은 성능을 보임
- Hybrid Retriever: Sparse와 Dense의 가중 평균 방식이 가장 우수한 성능을 보임 (Top 5에서 90% 정확도)

### Reader

- 최종 선택 모델: klue/roberta-large
- 성능: EM = 71.67, F1 = 80.55 (eval 기준)

### 앙상블

- Hard voting 방식이 Soft voting보다 더 좋은 성능을 보임


# 주요 학습 내용
- 데이터 분석 및 전처리의 중요성
- 다양한 retriever 및 reader 모델의 특성과 성능 비교
- 하이퍼파라미터 튜닝의 중요성
- 앙상블 기법을 통한 성능 향상

# 향후 개선 방향
- 더 체계적인 실험 설계 및 가설 검증 프로세스 구축
- 비정형 데이터에 대한 더 깊이 있는 EDA 수행
- 데이터 증강 기법의 개선 및 적용

# 참고 문헌
BERT : https://arxiv.org/abs/1810.04805 \
DPR : https://arxiv.org/abs/2004.04906 \
KLUE : https://arxiv.org/abs/2105.09680 \
Using the Hammer Only on Nails: A Hybrid Method for Evidence Retrieval for Question Answering: https://arxiv.org/abs/2009.10791
