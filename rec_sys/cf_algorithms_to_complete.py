# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np

def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None


def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    # dot = complete_code("fast_cosine_sim")
    dot = np.dot(um_normalized, vector)
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]

        # print("user index:")
        # print(user_index)
        # print("Item index:")
        # print(item_index)
        # print("orig_utility_matrix")
        # print(orig_utility_matrix)
        # print("clean_utility_matrix")
        # print(clean_utility_matrix)
        # print("Users who rated")
        # print(users_who_rated)
        # print("Similarities")
        # print(similarities)
        # print("Fast cosine sim of user col")
        # print(fast_cosine_sim(user_col, user_col))

        # print()
        # print("neighborhood size:")
        # print(neighborhood_size)
        # print("My answer:")
        # print(similarities[users_who_rated])

        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        # best_among_who_rated = complete_code("users with highest similarity")
        # Making rows out of [user's similarity, user] in order to sort for highest similarity
        best_among_who_rated = np.dstack((similarities[users_who_rated], np.arange(users_who_rated.size)))
        print("After stack")
        print(best_among_who_rated[0])
        # Truncating down list from 3d to 2d
        best_among_who_rated = best_among_who_rated[0]
        # Sorting the list by similarities
        best_among_who_rated = best_among_who_rated[best_among_who_rated[:, 0].argsort()]
        # Pulling out the list of users now sorted by their similarity
        best_among_who_rated = best_among_who_rated[:, 1]
        # Need to cast to int
        best_among_who_rated = best_among_who_rated.astype(int)

        print("After argsort")
        print(best_among_who_rated)

        print("HERE")
        print(best_among_who_rated)

        # best_among_who_rated = np.array([1, 0, 2])
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]

        print("after first best change")
        print(best_among_who_rated)
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        print("after 2nd best change")
        print(best_among_who_rated)

        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]

        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            rating_of_item = complete_code("compute the ratings")
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

