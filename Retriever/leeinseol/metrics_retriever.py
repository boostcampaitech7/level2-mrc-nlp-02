def recall_k(ground_truth_id, preds, k): 
    # ground_truth_id : 1-dim list(num_docs)
    # indices : 2-dim list(num_docs, top_k)
    assert k <= len(preds[0]), "recall k must same or smaller then top_k." 
    assert k >= 0, "recall k must bigger then 0."
    
    results = []
    num_docs = len(ground_truth_id)

    for i in range(num_docs):
        # 현재 문서의 ground truth id가 상위 k개 내에 존재하는지 확인
        if ground_truth_id[i] in preds[i][:k]:
            results.append(1)
        else :
            results.append(0)

    # recall@k 계산 (전체 문서 중에서 상위 k개 내에 정답이 있는 경우의 비율)
    return results

def mrr_k(ground_truth_id, preds, k) :
    # ground_truth_id : 1-dim list(num_docs)
    # indices : 2-dim list(num_docs, top_k) 
    assert k <= len(preds[0]), "mrr k must small or same with top_k." 
    assert k >= 0, "mrr k must bigger then 0."

    ranks = []
    for true, pred in zip(ground_truth_id, preds) :
        try :
            rank = pred[:k].index(true) + 1 # 1-based index
            ranks.append(1/rank)
        except ValueError :
            ranks.append(0) # 관련 문서가 검색 결과에 없는 경우 0으로 처리 
    return ranks