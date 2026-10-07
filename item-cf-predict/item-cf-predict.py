def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    r_t = 0
    s = 0
    for i in range(len(user_ratings)):
        if i != target and user_ratings[i] > 0 and item_similarities[i] > 0:
            r_t += item_similarities[i]*user_ratings[i]
            s += item_similarities[i]
    if s == 0:
        return 0
    else:
        return r_t/s