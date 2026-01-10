# 🎬 Movie Recommender System (Collaborative Filtering)

A machine learning based recommendation engine built with **TensorFlow** and **NumPy**. This system uses **Collaborative Filtering** (Low Rank Matrix Factorization) to predict user ratings for movies they haven't seen yet, based on their existing preferences and the collective behavior of other users.

## 🚀 Key Features

* **Collaborative Filtering:** Learns latent features for both users and movies simultaneously without needing manual genre tagging.
* **Vectorized Implementation:** Uses TensorFlow matrix operations (`tf.matmul`) instead of slow Python loops, allowing for training on thousands of movies in seconds.
* **Mean Normalization:** Implements advanced preprocessing to handle the "Cold Start" problem (ensuring reasonable predictions for new users or unrated movies).
* **Customizable User Profile:** Allows you to input your own movie ratings to generate personalized recommendations immediately.
* **Robust Data Handling:** Correctly handles sparse matrices and missing data (`NaNs`) using a hybrid masking approach.

## 🛠️ Tech Stack

* **Python 3.x**
* **TensorFlow:** For gradient descent and automatic differentiation.
* **NumPy:** For high-performance matrix manipulation.
* **Pandas:** For loading and processing the CSV datasets.

## 📂 Dataset

This project uses the **MovieLens "Small" Dataset**, which contains:
* 100,000 ratings
* 9,000 movies
* 600 users

*Dataset Source: [MovieLens / GroupLens Research](https://grouplens.org/datasets/movielens/latest/)*

## 🧠 How It Works

### 1. Matrix Factorization
The system tries to predict a rating $y^{(i,j)}$ for movie $i$ by user $j$ using the formula:
$$y^{(i,j)} = \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} + \mu_i$$

Where:
* $\mathbf{w}^{(j)}$: Parameters for user $j$ (user's taste).
* $\mathbf{x}^{(i)}$: Features for movie $i$ (movie's genre/style).
* $b^{(j)}$: User bias.
* $\mu_i$: Average rating of movie $i$.

### 2. The Cost Function
The model learns $\mathbf{w}$, $\mathbf{x}$, and $b$ by minimizing the **Mean Squared Error (MSE)** between predictions and actual ratings, with **L2 Regularization** to prevent overfitting:

$$J = \frac{1}{2} \sum_{(i,j):r(i,j)=1} ((\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}) - y^{(i,j)})^2 + \frac{\lambda}{2} (\sum \mathbf{w}^2 + \sum \mathbf{x}^2)$$

## ⚡ How to Run

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas tensorflow
    ```

2.  **Download Data:**
    Ensure you have `movies.csv` and `ratings.csv` in the same directory as the script.

3.  **Run the Script:**
    ```bash
    python main.py
    ```

### Customizing Your Ratings
To get recommendations for *yourself*, open the script and modify the `my_ratings` section:

```python
# Rate movies using their Matrix Row Index (or map via Movie ID)
# Example: Rating 'Toy Story' (ID 1) as 5 stars
my_ratings[0] = 5.0 
my_ratings[100] = 4.5


📊 Sample Output
After training for ~200 iterations, the system outputs personalized predictions:
Top recommendations for you:
Predicting 5.00 for Babe (1995)
Predicting 5.00 for Lamerica (1994)
Predicting 5.00 for Heidi Fleiss: Hollywood Madam (1995)
Predicting 5.00 for Taxi Driver (1976)
Predicting 5.00 for Awfully Big Adventure, An (1995)
Predicting 5.00 for Clerks (1994)
Predicting 5.00 for Hoop Dreams (1994)
Predicting 5.00 for Like Water for Chocolate (Como agua para chocolate) (1992)
Predicting 5.00 for Three Colors: Red (Trois couleurs: Rouge) (1994)
Predicting 5.00 for Adventures of Priscilla, Queen of the Desert, The (1994)