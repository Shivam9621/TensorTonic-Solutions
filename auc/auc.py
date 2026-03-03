def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have same length")
    
    total_area = 0.0
    
    for i in range(len(fpr) - 1):
        width = fpr[i + 1] - fpr[i]
        height = (tpr[i] + tpr[i + 1]) / 2
        total_area += width * height
    
    return total_area