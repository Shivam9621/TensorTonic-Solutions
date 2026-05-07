import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    
    if not recommendations:
        return 0.0
    
    total_novelty = 0
    
    for item in recommendations:
        popularity = item_counts[item] / n_users
        total_novelty += -math.log2(popularity)
    
    return total_novelty / len(recommendations)