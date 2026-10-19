def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    prec, rec = 0, 0
    for x in recommended[:k]:
        if x in relevant:
            prec +=1
            rec += 1
    return [prec/k, rec/len(relevant)]