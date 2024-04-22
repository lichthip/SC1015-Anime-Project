# SC1015-Anime-Project
# What makes a good anime?

**Introduction to Data Science and Artificial Intelligence
Lab FDDB Team 7 Mini-Project**

# Summary of the Project

We wanted to analyse what makes a "good" anime. Using the average community rating for an anime on website MyAnimeList as an estimate of the quality of an anime, we try to predict the rating for an anime using predictor variables from the dataset. 

Link to dataset on Kaggle: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database.
Raw dataset extracted from Kaggle [here](Datasets/anime.csv).

### Tools we used
We used Python in Jupyter Notebook, and the following libraries:
 - NumPy
 - Pandas
 - Seaborn
 - Matplotlib.pyplot
 - scikit-learn


# Initial Data Processing

Code section described below available [here](Codes/Data_extraction_cleaning_and_EDA.ipynb).

## Data Cleaning

Upon checking the data types of the variables in the dataset, we found that the 'episodes' column was of type object instead of integer, due to 340 rows of anime having 'Unknown' as their episode value. To avoid skewing the distribution of variable 'episodes', we impute these missing values with a measure of central tendency: the mean value of episodes of the remaining anime. We could then convert the 'episodes' column to the int64 type.

To reduce the variables involved, we grouped the least common 'types' entries together under Others, leaving only 'TV', 'Movie', 'OVA' and 'Others' as valid entries under the column 'types'.

We also removed rows of anime with certain genres which reference inappropriate content for the purposes of this project.

After cleaning the data, we are left with [this](Datasets/anime_clean.csv).

## Exploratory Data Analysis

### Distribution of overall ratings

We visualised the overall distribution of anime ratings in a violin plot, which revealed that most anime have ratings falling between 5 -- 8, but with a significant number of outliers above and below. The number of outliers was deemed too significant for us to ignore and drop the rows.

### The 'types' column

Analysing the 'types' column, we looked at the number of anime corresponding to each type, visualised in a histogram. We then reviewed the violin plots of the distribution of ratings corresponding to each type, noting the differences in median, spread and extreme values. For instance, TV anime has the smallest spread of ratings. TV and Movie anime tended to have higher ratings than the other types on average, and their maximum values are higher as well.

### The 'episodes' column

To understand the 'episodes' column better, we plotted a histogram of the frequency of the number of episodes in an anime. To better show the number of anime with exceptionally large episode counts, we use a logarithmic scale in the y-axis, since they would otherwise be dwarfed by the vast majority of anime with less than 100 episodes. We then look at the relationship between the episode count and the community rating. The extremely long-running anime seem to have average ratings or better, which could be the reason for their longevity.

Code section described below available [here](Codes/Averages_Linear_Regression.ipynb).

### The 'genres' column

To understand the effect of individual genres on rating, we looked for anime with only each genre and took an average of the ratings of these anime. Doing this, we found that looking at single-genre anime, Martial Arts anime are on average the most well-received. At the same time, we realised that there were no anime with only one genre corresponding to the genres 'Police', 'Josei' and 'Super Power'. We then appended the mean value of *all* anime corresponding to these genres.

## Machine Learning

### Basic Linear Regression

Using the averages for each genre found previously, we replace the 'genres' value in each row with the equally-weighted average of the average ratings of its genres. Using the 'genres', 'types' and 'episodes' columns as predictors, we use linear regression to estimate the rating for each anime, with an 80--20 train--test split, performed on a fixed seed.

The altered data frame used for this can be found [here](Datasets/anime_clean_linear_regression.csv).

The performance of this model was rather poor, with an adjusted R<sup>2</sup> squared Error (MSE) of the model was ~0.9339 on the train set and ~0.9363 on the test set, with units of rating squared. The small MSE value in spite of the poor performance of the model can be explained by the significance of a change in 1 unit of rating. Looking at the raw error, we see that majority of the errors have magnitude between 0 and 1 compared with those with magnitude larger than 1. As a result, the effect of majority of points outweighs those much poorer predictions in the MSE.

Code section described below available [here](Codes/One_Hot_Encoding_Lasso_Ridge_ElasticNet_Regression.ipynb).

### More advanced models

We decided to use more advanced regression models to try to achieve better prediction performance.

However, to achieve this, we first had to properly consider the effect of each genre separately, instead of using the aforementioned average of averages. Thus, we **one-hot encoded** each genre in 'genres', creating a new column for each genre, assigning a '1' in each row if the anime in that row had the corresponding genre and a '0' otherwise, then checking that these columns were of data type int64.

The modified data after the one-hot encoding process can be found [here](Datasets/anime_clean_encoded.csv).

### Regularised Linear Regressions

Thereafter we experimented with regularised linear regressions. These are types of linear regressions which penalise the parameters by reducing their impact on the response variable in a certain way to reduce overfitting.

In L1 regularisation, also known as **Lasso** regression, the least important variables are reduced to having coefficients of zero. The number of affected variables is controlled by the L1 hyperparameter. This simplifies the model by eliminating its dependence on variables that are least significant, which could reduce overfitting.

In L2 regularisation, also known as **Ridge** regression, the parameters are penalised in proportion to their sum of squares, so larger coefficients will be more heavily penalised than smaller ones. The multiplier of the penalty term is determined by the L2 hyperparameter. It seeks to reduce overfitting by reducing high coefficients which can arise due to multicollinearity.

Between the Lasso and Ridge regressions is the **Elastic Net** regression, which penalises its parameters using a linear combination of the L1 and L2 hyperparameters, the proportion of which is yet another hyperparameter.

We tuned the hyperparameters so that we could get the most optimal model of each regularised regression method using GridSearchCV, once again split into the same train--test set as used in the basic linear regression. 

After finding the best hyperparameters, we compared the performance of each model. The model performance across the board was much better than the basic linear regression, with adjusted R^2^ of Lasso at ~0.2734, Ridge at ~0.2705, and Elastic Net at ~0.2718, of which the best model would be the one using Lasso regression.

## Analysis and Insights

We printed the coefficients for all 37 columns in descending order for all 3 models. We found that the 'Josei' genre, which means the anime is targeted at adult women, had the most positive effect on an anime's ratings by some margin across the 3 models. The next 2 predictors with the most positive effect were the 'Mystery' and 'Drama' genres. Of the types of anime, being of the 'TV' type was the only type to have a positive influence on ratings. The more episodes an anime has, the greater the positive effect on its ratings. Thus, using our insights, we can make a prediction of how the community on MyAnimeList will perceive an anime with a given set of genres, types and episodes.

Surprisingly, despite the high average rating in single-genre anime, the 'Martial Arts' genre was estimated to have little effect on an anime's ratings under these more advanced models. This could be due to a discrepancy in the ratings for anime with only the Martial Arts genre and anime with a combination of Martial Arts and other genres.

We had initially hypothesised that more common genres of anime such as 'Action' and 'Adventure' would have a more positive effect on ratings. They do still have a positive effect on ratings, but only rank 20^th^ and 22^nd^ respectively under the Lasso model, for example. This could be due to the sheer amount of anime with these genres causing a wide spread of quality and thus ratings.

## Limitations of our Analysis

Our best 3 models only achieved an adjusted R^2^ of slightly over 0.27. This is likely due to the size of the dataset and the outliers in ratings, or perhaps reflect an underlying bias or variance that was not captured.

We excluded the 'members' column from use in creating our models as its effect on model performance was significantly detrimental. This could cause some bias in our models as anime with a larger community of followers may be rated more highly.

In addition, beyond the scope of the dataset, there may be other abstract factors that are difficult to quantify affecting the ratings of an anime, for instance the art style used.

We are also unable to capture the fluctuations in rating of anime within this model, which could affect model performance as time passes and the community's tastes and preferences in anime shift.





