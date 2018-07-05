
# k-means Clustering of Movie Ratings

Say you're a data analyst at Netflix and you want to explore the similarities and differences in people's tastes in movies based on how they rate different movies. Can understanding these ratings contribute to a movie recommendation system for users? Let's dig into the data and see.

The data we'll be using comes from the wonderful [MovieLens](https://movielens.org/) [user rating dataset](https://grouplens.org/datasets/movielens/). We'll be looking at individual movie ratings later in the notebook, but let us start with how ratings of genres compare to each other.

## Dataset overview
The dataset has two files. We'll import them both into pandas dataframes:


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import helper

# Import the Movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
movies.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Import the ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>



Now that we know the structure of our dataset, how many records do we have in each of these tables?


```python
print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')
```

    The dataset contains:  100004  ratings of  9125  movies.



## Romance vs. Scifi
Let's start by taking a subset of users, and seeing what their preferred genres are. We're hiding the most data preprocessing in helper functions so the focus is on the topic of clustering. It would be useful if you skim helper.py to see how these helper functions are implemented after finishing this notebook.


```python
# Calculate the average rating of romance and scifi movies

genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
genre_ratings.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_romance_rating</th>
      <th>avg_scifi_rating</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.50</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.59</td>
      <td>3.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.65</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.50</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.08</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>



The function `get_genre_ratings` calculated each user's average rating of all romance movies and all scifi movies. Let's bias our dataset a little by removing people who like both scifi and romance, just so that our clusters tend to define them as liking one genre more than the other.


```python
biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

print( "Number of records: ", len(biased_dataset))
biased_dataset.head()
```

    Number of records:  183





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>avg_romance_rating</th>
      <th>avg_scifi_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.50</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3.65</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>2.90</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>2.93</td>
      <td>3.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>2.89</td>
      <td>2.62</td>
    </tr>
  </tbody>
</table>
</div>



So we can see we have 183 users, and for each user we have their average rating of the romance and sci movies they've watched.

Let us plot this dataset:


```python
%matplotlib inline

helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')
```


![png](output_10_0.png)


We can see come clear bias in this sample (that we created on purpose). How would it look if we break the sample down into two groups using k-means?


```python
# Let's turn our dataset into a list
X = biased_dataset[['avg_scifi_rating','avg_romance_rating']].values
```

* Import [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* Prepare KMeans with n_clusters = 2
* Pass the dataset **X** to KMeans' fit_predict method and retrieve the clustering labels into *predictions*


```python
# TODO: Import KMeans
from sklearn.cluster import KMeans

# TODO: Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)

# TODO: use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X)

# Plot
helper.draw_clusters(biased_dataset, predictions)
```


![png](output_14_0.png)


We can see that the groups are mostly based on how each person rated romance movies. If their average rating of romance movies is over 3 stars, then they belong to one group. Otherwise, they belong to the other group.

What would happen if we break them down into three groups?


```python

# TODO: Create an instance of KMeans to find three clusters
kmeans_2 = KMeans(n_clusters=3)

# TODO: use fit_predict to cluster the dataset
predictions_2 = kmeans_2.fit_predict(X)

# Plot
helper.draw_clusters(biased_dataset, predictions_2)
```


![png](output_16_0.png)


Now the average scifi rating is starting to come into play. The groups are:
 * people who like romance but not scifi
 * people who like scifi but not romance
 * people who like both scifi and romance
 
Let's add one more group


```python
# TODO: Create an instance of KMeans to find four clusters
kmeans_3 = KMeans(n_clusters=4)

# TODO: use fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(X)

# Plot
helper.draw_clusters(biased_dataset, predictions_3)
```


![png](output_18_0.png)


We can see that the more clusters we break our dataset down into, the more similar the tastes of the population of each cluster to each other.

## Choosing K
Great, so we can cluster our points into any number of clusters. What's the right number of clusters for this dataset?

There are [several](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) ways of choosing the number of clusters, k. We'll look at a simple one called "the elbow method". The elbow method works by plotting the ascending values of k versus the total error calculated using that k. 

How do we calculate total error?
One way to calculate the error is squared error. Say we're calculating the error for k=2. We'd have two clusters each having one "centroid" point. For each point in our dataset, we'd subtract its coordinates from the centroid of the cluster it belongs to. We then square the result of that subtraction (to get rid of the negative values), and sum the values. This would leave us with an error value for each point. If we sum these error values, we'd get the total error for all points when k=2.

Our mission now is to do the same for each k (between 1 and, say, the number of elements in our dataset)


```python
# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)

# Calculate error values for all k values we're interested in
errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

```


```python
# Optional: Look at the values of K vs the silhouette score of running K-means with that value of k
list(zip(possible_k_values, errors_per_k))
```




    [(2, 0.35588178764728268),
     (7, 0.37251245002986361),
     (12, 0.36053028133753795),
     (17, 0.36133053950678817),
     (22, 0.35918256485645172),
     (27, 0.33838730891653518),
     (32, 0.37507027867176423),
     (37, 0.37568739612958918),
     (42, 0.37424007948089139),
     (47, 0.36881833045340529),
     (52, 0.3616091192785712),
     (57, 0.37516850181963052),
     (62, 0.35933029980676351),
     (67, 0.34988341071694795),
     (72, 0.33967298261751322),
     (77, 0.34236352651196883),
     (82, 0.33007244845279948),
     (87, 0.32667028562629674),
     (92, 0.33921364612198379),
     (97, 0.33337051692447206),
     (102, 0.29602264046443116),
     (107, 0.30192783029869558),
     (112, 0.29433267136490193),
     (117, 0.27084543797315741),
     (122, 0.27189792081715225),
     (127, 0.24924922967084562),
     (132, 0.24542882874268315),
     (137, 0.22801623230853793),
     (142, 0.21994240424860978),
     (147, 0.1988012278534107),
     (152, 0.1811203933202615),
     (157, 0.16447514022085591),
     (162, 0.14759705165906709),
     (167, 0.12820427579529109),
     (172, 0.10075966098920461),
     (177, 0.064230120163174503),
     (182, 0.054644808743169397)]




```python
# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlabel('K - number of clusters')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')
```


![png](output_22_0.png)


Looking at this graph, good choices for k include 7, 22, 27, 32, amongst other values (with a slight variation between different runs). Increasing the number of clusters (k) beyond that range starts to result in worse clusters (according to Silhouette score)

My pick would be k=7 because it's easier to visualize:


```python
# TODO: Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7)

# TODO: use fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# plot
helper.draw_clusters(biased_dataset, predictions_4, cmap='Accent') 
```


![png](output_24_0.png)


Note: As you try to plot larger values of k (more than 10), you'll have to make sure your plotting library is not reusing colors between clusters. For this plot, we had to use the [matplotlib colormap](https://matplotlib.org/examples/color/colormaps_reference.html) 'Accent' because other colormaps either did not show enough contrast between colors, or were recycling colors past 8 or 10 clusters.


## Throwing some Action into the mix
So far, we've only been looking at how users rated romance and scifi movies. Let's throw another genre into the mix. Let's add the Action genre.

Our dataset now looks like this:


```python
biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies, 
                                                     ['Romance', 'Sci-Fi', 'Action'], 
                                                     ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()
```

    Number of records:  183





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>avg_romance_rating</th>
      <th>avg_scifi_rating</th>
      <th>avg_action_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.50</td>
      <td>2.40</td>
      <td>2.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3.65</td>
      <td>3.14</td>
      <td>3.47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>2.90</td>
      <td>2.75</td>
      <td>3.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>2.93</td>
      <td>3.36</td>
      <td>3.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>2.89</td>
      <td>2.62</td>
      <td>3.21</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_with_action = biased_dataset_3_genres[['avg_scifi_rating',
                                         'avg_romance_rating', 
                                         'avg_action_rating']].values
```


```python
# TODO: Create an instance of KMeans to find seven clusters
kmeans_5 = KMeans(n_clusters=7)

# TODO: use fit_predict to cluster the dataset
predictions_5 = kmeans_5.fit_predict(X)

# plot
helper.draw_clusters_3d(biased_dataset_3_genres, predictions_5)
```


![png](output_29_0.png)


We're still using the x and y axes for scifi and romance respectively. We are using the size of the dot to roughly code the 'action' rating (large dot for avg ratings over than 3, small dot otherwise).

We can start seeing the added genre is changing how the users are clustered. The more data we give to k-means, the more similar the tastes of the people in each group would be. Unfortunately, though, we lose the ability to visualize what's going on past two or three dimensions if we continue to plot it this way. In the next section, we'll start using a different kind of plot to be able to see clusters with up to fifty dimensions.

## Movie-level Clustering
Now that we've established some trust in how k-means clusters users based on their genre tastes, let's take a bigger bite and look at how users rated individual movies. To do that, we'll shape the dataset in the form of userId vs user rating for each movie. For example, let's look at a subset of the dataset:


```python
# Merge the two tables then pivot so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example:')
user_movie_ratings.iloc[:6, :10]
```

    dataset dimensions:  (671, 9064) 
    
    Subset example:





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>"Great Performances" Cats (1998)</th>
      <th>$9.99 (2008)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Neath the Arizona Skies (1934)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The dominance of NaN values presents the first issue. Most users have not rated and watched most movies. Datasets like this are called "sparse" because only a small number of cells have values. 

To get around this, let's sort by the most rated movies, and the users who have rated the most number of movies. That will present a more 'dense' region when we peak at the top of the dataset.

If we're to choose the most-rated movies vs users with the most ratings, it would look like this:


```python
n_movies = 30
n_users = 18
most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)

print('dataset dimensions: ', most_rated_movies_users_selection.shape)
most_rated_movies_users_selection.head()
```

    dataset dimensions:  (18, 30)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>Forrest Gump (1994)</th>
      <th>Pulp Fiction (1994)</th>
      <th>Shawshank Redemption, The (1994)</th>
      <th>Silence of the Lambs, The (1991)</th>
      <th>Star Wars: Episode IV - A New Hope (1977)</th>
      <th>Jurassic Park (1993)</th>
      <th>Matrix, The (1999)</th>
      <th>Toy Story (1995)</th>
      <th>Schindler's List (1993)</th>
      <th>Terminator 2: Judgment Day (1991)</th>
      <th>...</th>
      <th>Dances with Wolves (1990)</th>
      <th>Fight Club (1999)</th>
      <th>Usual Suspects, The (1995)</th>
      <th>Seven (a.k.a. Se7en) (1995)</th>
      <th>Lion King, The (1994)</th>
      <th>Godfather, The (1972)</th>
      <th>Lord of the Rings: The Fellowship of the Ring, The (2001)</th>
      <th>Apollo 13 (1995)</th>
      <th>True Lies (1994)</th>
      <th>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>508</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>653</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



That's more like it. Let's also establish a good way for visualizing these ratings so we can attempt to visually recognize the ratings (and later, clusters) when we look at bigger subsets.

Let's use colors instead of the number ratings:


```python
helper.draw_movies_heatmap(most_rated_movies_users_selection)
```


![png](output_35_0.png)


Each column is a movie. Each row is a user. The color of the cell is how the user rated that movie based on the scale on the right of the graph.

Notice how some cells are white? This means the respective user did not rate that movie. This is an issue you'll come across when clustering in real life. Unlike the clean example we started with, real-world datasets can often be sparse and not have a value in each cell of the dataset. This makes it less straightforward to cluster users directly by their movie ratings as k-means generally does not like missing values.

For performance reasons, we'll only use ratings for 1000 movies (out of the 9000+ available in the dataset).


```python
user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)
```

To have sklearn run k-means clustering to a dataset with missing values like this, we will first cast it to the [sparse csr matrix](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.html) type defined in the SciPi library. 

To convert from a pandas dataframe to a sparse matrix, we'll have to convert to SparseDataFrame, then use pandas' `to_coo()` method for the conversion.

Note: `to_coo()` was only added in later versions of pandas. If you run into an error with the next cell, make sure pandas is up to date.


```python
sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())
```

## Let's cluster!
With k-means, we have to specify k, the number of clusters. Let's arbitrarily try k=20 (A better way to pick k is as illustrated above with the elbow method. That would take some processing time to run, however.):


```python
# 20 clusters
predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)
```

To visualize some of these clusters, we'll plot each cluster as a heat map:


```python
max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
helper.draw_movie_clusters(clustered, max_users, max_movies)
```

    cluster # 16
    # of users in cluster: 297. # of users in plot: 70



![png](output_43_1.png)


    cluster # 17
    # of users in cluster: 81. # of users in plot: 70



![png](output_43_3.png)


    cluster # 12
    # of users in cluster: 10. # of users in plot: 10



![png](output_43_5.png)


    cluster # 13
    # of users in cluster: 34. # of users in plot: 34



![png](output_43_7.png)


    cluster # 15
    # of users in cluster: 53. # of users in plot: 53



![png](output_43_9.png)


    cluster # 0
    # of users in cluster: 16. # of users in plot: 16



![png](output_43_11.png)


    cluster # 18
    # of users in cluster: 37. # of users in plot: 37



![png](output_43_13.png)


    cluster # 7
    # of users in cluster: 50. # of users in plot: 50



![png](output_43_15.png)


    cluster # 10
    # of users in cluster: 15. # of users in plot: 15



![png](output_43_17.png)


    cluster # 5
    # of users in cluster: 15. # of users in plot: 15



![png](output_43_19.png)


    cluster # 9
    # of users in cluster: 17. # of users in plot: 17



![png](output_43_21.png)


    cluster # 19
    # of users in cluster: 12. # of users in plot: 12



![png](output_43_23.png)


There are several things to note here:
* The more similar the ratings in a cluster are, the more **vertical** lines in similar colors you'll be able to trace in that cluster. 
* It's super interesting to spot trends in clusters:
 * Some clusters are more sparse than others, containing people who probably watch and rate less movies than in other clusters.
 * Some clusters are mostly yellow and bring together people who really love a certain group of movies. Other clusters are mostly green or navy blue meaning they contain people who agree that a certain set of movoies deserves 2-3 stars.
 * Note how the movies change in every cluster. The graph filters the data to only show the most rated movies, and then sorts them by average rating.
 * Can you track where the Lord of the Rings movies appear in each cluster? What about Star Wars movies?
* It's easy to spot **horizontal** lines with similar colors, these are users without a lot of variety in their ratings. This is likely one of the reasons for Netflix switching from a stars-based ratings to a thumbs-up/thumbs-down rating. A rating of four stars means different things to different people.
* We did a few things to make the clusters visibile (filtering/sorting/slicing). This is because datasets like this are "sparse" and most cells do not have a value (because most people did not watch most movies). 

## Prediction
Let's pick a cluster and a specific user and see what useful things this clustering will allow us to do.

Let's first pick a cluster:


```python
# TODO: Pick a cluster ID from the clusters above
cluster_number = 9

# Let's filter to only see the region of the dataset with the most number of values 
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

cluster = helper.sort_by_rating_density(cluster, n_movies, n_users)
helper.draw_movies_heatmap(cluster, axis_labels=False)
```


![png](output_45_0.png)


And the actual ratings in the cluster look like this:


```python
cluster.fillna('').head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Forrest Gump (1994)</th>
      <th>Back to the Future (1985)</th>
      <th>Ferris Bueller's Day Off (1986)</th>
      <th>Star Wars: Episode I - The Phantom Menace (1999)</th>
      <th>Indiana Jones and the Last Crusade (1989)</th>
      <th>Mrs. Doubtfire (1993)</th>
      <th>E.T. the Extra-Terrestrial (1982)</th>
      <th>Lion King, The (1994)</th>
      <th>Independence Day (a.k.a. ID4) (1996)</th>
      <th>Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)</th>
      <th>...</th>
      <th>Rumble in the Bronx (Hont faan kui) (1995)</th>
      <th>Raising Arizona (1987)</th>
      <th>Star Trek: First Contact (1996)</th>
      <th>Scarface (1983)</th>
      <th>Good Morning, Vietnam (1987)</th>
      <th>Cinderella (1950)</th>
      <th>Miss Congeniality (2000)</th>
      <th>Basic Instinct (1992)</th>
      <th>So I Married an Axe Murderer (1993)</th>
      <th>Old School (2003)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>...</td>
      <td></td>
      <td>3</td>
      <td>2.5</td>
      <td></td>
      <td></td>
      <td></td>
      <td>2.5</td>
      <td>2.5</td>
      <td>3.5</td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td>4.5</td>
      <td></td>
      <td>4</td>
      <td></td>
      <td></td>
      <td>3.5</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>...</td>
      <td></td>
      <td>3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>4.5</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>4.0</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td>3</td>
      <td>3.5</td>
      <td>4</td>
      <td></td>
      <td>3</td>
      <td>2</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.5</td>
      <td>3.5</td>
      <td>...</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td>3.5</td>
      <td>3</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 300 columns</p>
</div>



Pick a blank cell from the table. It's blank because that user did not rate that movie. Can we predict whether she would like it or not? Since the user is in a cluster of users that seem to have similar taste, we can take the average of the votes for that movie in this cluster, and that would be a reasonable predicition for much she would enjoy the film.


```python
# TODO: Fill in the name of the column/movie. e.g. 'Forrest Gump (1994)'
# Pick a movie from the table above since we're looking at a subset
movie_name = 'Forrest Gump (1994)'

cluster[movie_name].mean()
```




    4.1764705882352944



And this would be our prediction for how she'd rate the movie.

## Recommendation
Let's reiterate what we did in the previous step. We have used k-means to cluster users according to their ratings. This lead us to clusters of users with similar ratings and thus generally a similar taste in movies. Based on this, when one user did not have a rating for a certain movie  we averaged the ratings of all the other users in the cluster, and that was our guess to how this one user would like the movie.

Using this logic, if we calculate the average score in this cluster for every movie, we'd have an understanding for how this 'taste cluster' feels about each movie in the dataset. 



```python
# The average rating of 20 movies as rated by the users in the cluster
cluster.mean().head(20)
```




    Forrest Gump (1994)                                                               4.176471
    Back to the Future (1985)                                                         4.117647
    Ferris Bueller's Day Off (1986)                                                   4.323529
    Star Wars: Episode I - The Phantom Menace (1999)                                  3.205882
    Indiana Jones and the Last Crusade (1989)                                         4.147059
    Mrs. Doubtfire (1993)                                                             3.176471
    E.T. the Extra-Terrestrial (1982)                                                 3.735294
    Lion King, The (1994)                                                             3.588235
    Independence Day (a.k.a. ID4) (1996)                                              3.294118
    Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)    4.323529
    Big (1988)                                                                        3.852941
    Jurassic Park (1993)                                                              3.882353
    Matrix, The (1999)                                                                4.500000
    Braveheart (1995)                                                                 4.000000
    Toy Story (1995)                                                                  3.812500
    Star Wars: Episode V - The Empire Strikes Back (1980)                             4.593750
    Star Wars: Episode IV - A New Hope (1977)                                         4.375000
    Mask, The (1994)                                                                  3.156250
    Groundhog Day (1993)                                                              3.968750
    Speed (1994)                                                                      3.406250
    dtype: float64



This becomes really useful for us because we can now use it as a recommendation engine that enables our users to discover movies they're likely to enjoy.

When a user logs in to our app, we can now show them recommendations that are appropriate to their taste. The formula for these recommendations is to select the cluster's highest-rated movies that the user did not rate yet.



```python
# TODO: Pick a user ID from the dataset
# Look at the table above outputted by the command "cluster.fillna('').head()" 
# and pick one of the user ids (the first column in the table)
user_id = 12

# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]

# Which movies did they not rate? (We don't want to recommend movies they've already rated)
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

# Let's sort by rating so the highest rated movies are presented first
avg_ratings.sort_values(ascending=False)[:20]
```




    Monty Python and the Holy Grail (1975)                 4.541667
    Eternal Sunshine of the Spotless Mind (2004)           4.285714
    Young Frankenstein (1974)                              4.285714
    Brazil (1985)                                          4.166667
    Big Fish (2003)                                        4.166667
    Sin City (2005)                                        4.166667
    Shrek 2 (2004)                                         4.150000
    Muppet Movie, The (1979)                               4.083333
    Hitchhiker's Guide to the Galaxy, The (2005)           4.071429
    Monsters, Inc. (2001)                                  4.038462
    Fish Called Wanda, A (1988)                            3.954545
    Twelve Monkeys (a.k.a. 12 Monkeys) (1995)              3.950000
    Labyrinth (1986)                                       3.928571
    Old School (2003)                                      3.916667
    Good Morning, Vietnam (1987)                           3.916667
    Star Wars: Episode III - Revenge of the Sith (2005)    3.909091
    Caddyshack (1980)                                      3.857143
    Blazing Saddles (1974)                                 3.812500
    Beautiful Mind, A (2001)                               3.785714
    Star Trek II: The Wrath of Khan (1982)                 3.785714
    Name: 0, dtype: float64



And these are our top 20 recommendations to the user!

### Quiz:
 * If the cluster had a movie with only one rating. And that rating was 5 stars. What would the average rating of the cluster for that movie be? How does that effect our simple recommendation engine? How would you tweak the recommender to address this issue?

## More on Collaborative Filtering
* This is a simplistic recommendation engine that shows the most basic idea of "collaborative filtering". There are many heuristics and methods to improve it. [The Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) tried to push the envelope in this area by offering a prize of US$1,000,000 to the recommendation algorithm that shows the most improvement over Netflix's own recommendation algorithm.
* That prize was granted in 2009 to a team called "BellKor's Pragmatic Chaos". [This paper](http://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf) shows their approach which employed an ensemble of a large number of methods. 
* [Netflix did not end up using this \$1,000,000 algorithm](https://thenextweb.com/media/2012/04/13/remember-netflixs-1m-algorithm-contest-well-heres-why-it-didnt-use-the-winning-entry/) because their switch to streaming gave them a dataset that's much larger than just movie ratings -- what searches did the user make? What other movies did the user sample in this session? Did they start watching a movie then stop and switch to a different movie? These new data points offered a lot more clues than the ratings alone.

## Take it Further

* This notebook showed user-level recommendations. We can actually use the almost exact code to do item-level recommendations. These are recommendations like Amazon's "Customers who bought (or viewed or liked) this item also bought (or viewed or liked)". These would be recommendations we can show on each movie's page in our app. To do this, we simple transpose the dataset to be in the shape of Movies X Users, and then cluster the movies (rather than the users) based on the correlation of their ratings.
* We used the smallest of the datasets Movie Lens puts out. It has 100,000 ratings. If you want to dig deeper in movie rating exploration, you can look at their [Full dataset](https://grouplens.org/datasets/movielens/) containing 24 million ratings.


