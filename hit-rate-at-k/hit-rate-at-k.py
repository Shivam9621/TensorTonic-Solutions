def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    
    n = len(recommendations)
    
    if n == 0:
        return 0.0
    
    hits = 0
    
    for recs, truth in zip(recommendations, ground_truth):
        top_k = set(recs[:k])
        
        if any(item in top_k for item in truth):
            hits += 1
    
    return hits / n