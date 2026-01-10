import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')


Y_df = ratings.pivot(index='movieId', columns='userId', values='rating')
Y_original = Y_df.to_numpy()


num_movies = Y_original.shape[0]
my_ratings = np.zeros(num_movies)

my_ratings[2700] = 4.5
my_ratings[2609] = 5.0
my_ratings[1206] = 3.0
my_ratings[1193] = 4.0      
my_ratings[656]  = 3.5
my_ratings[318]  = 4.0          
my_ratings[50]   = 4.0
my_ratings[1100] = 4.0
my_ratings[780]  = 5.0
my_ratings[430]  = 3.0
my_ratings[929]  = 5  
my_ratings[246]  = 5  
my_ratings[2716] = 3  
my_ratings[1150] = 5  
my_ratings[382]  = 2   
my_ratings[366]  = 5   
my_ratings[622]  = 5  
my_ratings[988]  = 3   
my_ratings[2925] = 1   
my_ratings[2937] = 4.5   
my_ratings[793]  = 2   


my_ratings_with_nans = np.where(my_ratings == 0, np.nan, my_ratings)


Y_combined = np.column_stack((my_ratings_with_nans, Y_original))


Y_mean = np.nanmean(Y_combined, axis=1).reshape(-1, 1)


R = (~np.isnan(Y_combined)).astype(int)


Y_clean = np.nan_to_num(Y_combined)


Y_norm = Y_clean - (Y_mean * R)


num_users = Y_combined.shape[1]
num_features = 50

tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users,  num_features), dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64),  name='b')

optimizer = keras.optimizers.Adam(learning_rate=1e-1)


def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    
    
    P = tf.matmul(X, tf.transpose(W)) + b
    
    
    j = (P - Y) * R 
    
    
    J = 0.5 * tf.reduce_sum(j**2)
    
    
    J += (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    
    return J


iterations = 200 
lambda_ = 1

print("Training...")
for iter in range(iterations):
    
    with tf.GradientTape() as tape:
        
        cost_value = cofi_cost_func_v(X, W, b, Y_norm, R, lambda_)

    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 20 == 0:
        print(f"Iteration {iter}: Cost {cost_value:0.1f}")

# Save the trained tensors to disk
np.save('W.npy', W.numpy())
np.save('X.npy', X.numpy())
np.save('b.npy', b.numpy())

p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()


pm = p + Y_mean


my_predictions = pm[:, 0]
my_predictions = np.clip(my_predictions, 0.5, 5.0)


ix = tf.argsort(my_predictions, direction='DESCENDING')


my_rated_indices = [i for i, x in enumerate(my_ratings) if x > 0]

print('\nTop recommendations for you:')
count = 0

for i in range(len(ix)):
    row_idx = ix[i]
    
    
    if row_idx not in my_rated_indices:
        
        
        # New (Fixed)
        actual_movie_id = Y_df.index[row_idx.numpy()]
        
        
        matches = movies.loc[movies['movieId'] == actual_movie_id, 'title']
        
        if len(matches) > 0:
            title = matches.values[0]
            print(f'Predicting {my_predictions[row_idx]:0.2f} for {title}')
            
            count += 1
            if count == 10: break