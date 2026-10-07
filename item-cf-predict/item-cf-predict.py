def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    num = 0
    den = 0
    for i in range(len(user_ratings)):
        if i != target and user_ratings[i] > 0 and item_similarities[i] > 0:
            num += item_similarities[i]*user_ratings[i]
            den += item_similarities[i]
    if den == 0:
        return 0
    else:
        return num/den