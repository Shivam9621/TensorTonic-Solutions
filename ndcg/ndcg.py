import math

def ndcg(relevance_scores, k):
    # Use only top k elements
    k = min(k, len(relevance_scores))
    
    def dcg(scores):
        total = 0.0
        for i in range(len(scores)):
            rel = scores[i]
            total += (2**rel - 1) / math.log2(i + 2)  # i+2 because index starts at 0
        return total
    
    # DCG for given ranking
    dcg_k = dcg(relevance_scores[:k])
    
    # IDCG (ideal ranking → sorted descending)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg_k = dcg(ideal_scores[:k])
    
    # Avoid division by zero
    if idcg_k == 0:
        return 0.0
    
    return dcg_k / idcg_k