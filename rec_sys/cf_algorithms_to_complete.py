# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
import rec_sys.data_util as cfd
import rec_sys.config as ConfigLf
import shelve


def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None

# New function to handle vector-vector pairs that may have sparse data
# Using pearson correlation from slide deck 1 slide 36
def centered_cosine_sim(u, v):
  # center and set up sparse data to be used in calculations
  vector_u = center_and_nan_to_zero(u)
  vector_v = center_and_nan_to_zero(v)
  dot = np.dot(vector_v, vector_u)

  # calculate the denominator of the pearson correlation
  denominator_v = 0 
  denominator_u = 0
  for i in vector_v:
    denominator_v = denominator_v + (i)**2

  for j in vector_u:
    denominator_u = denominator_u + (j)**2

  denominator = denominator_u * denominator_v

  return dot / denominator



# New function to handle vector-matrix pairs that may have sparse data
def fast_centered_cosine_sim():
  return None

# Exercise 4, generate the python shelves rated_by[] and user_col[] from a dataset
# for rated_by need to store users who have rated an item id so the dict goes
# rated_by[item_id] = user_id
# then user_col is a user and the movie ids they have reviewd
def generate_shevles():
  ratings_tf, user_ids_voc, movie_ids_voc = cfd.load_movielens_tf(ConfigLf.ConfigLf)

  rated_by = shelve.open('rated_by_shelf', writeback=True)
  user_col = shelve.open('user_col_shelf', writeback=True)

  ratings = list(ratings_tf.as_numpy_iterator())

  for i in ratings:
    rated_by[str(i['movie_id'])] = str(i['user_id'])

    user_col[str(i['user_id'])] = str(i['movie_id'])
  
  a = rated_by
  b = user_col
  rated_by.close()
  user_col.close()
  return a, b 
 

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

    # Have to truncate in case they do not match
    dot = um_normalized @ vector[:np.shape(um_normalized)[1]]
    # dot = np.dot(um_normalized, vector.reshape)

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

        # Truncating down list from 3d to 2d
        best_among_who_rated = best_among_who_rated[0]
        # Sorting the list by similarities
        best_among_who_rated = best_among_who_rated[best_among_who_rated[:, 0].argsort()]
        # Pulling out the list of users now sorted by their similarity
        best_among_who_rated = best_among_who_rated[:, 1]
        # Need to cast to int
        best_among_who_rated = best_among_who_rated.astype(int)

        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]

        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]

        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]

        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            # rating_of_item = complete_code("compute the ratings")
            rating_of_item = np.dot(np.sort(similarities)[-neighborhood_size:], best_among_who_rated) / np.sum(similarities)
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

