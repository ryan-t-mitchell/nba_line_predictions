# Project Airball 
The goal of this project is to make a profitable betting algorithm using NBA regular season data and inital point spreads across a series of betting sites.

Historical odds data (2015-16 through 2021-22 seasons) downloaded from: https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/tree/master/Odds-Data/Odds-Data-Clean

## Data Gathering Steps
1. First, run the final_team_stats.ipynb file, which pings the NBA API for relevant box score data which will feed into our final feature set.
2. Next, download historical odds data from the repo linked above (2015-16 through 2021-22 seasons).
3. Run the odds.ipynb file, which concatenates historical odds data into a single dataframe for processing. For 2022-23 and later seasons, scrape the sports betting sites and append the mode of the home team spread values.
4. Construct the training data set using the NBA API data and the odds data (construct_training_set.ipynb).

## Modeling
1. Our initial model (created in initial_model.ipynb) generates a random forest machine learning model using the HOME SPREAD from the betting sites, along with custom features that our team engineered. This initial model failed to outperform a model using just the HOME SPREAD. Perhaps unsurprisingly, the HOME SPREAD feature was by far the most important feature (feature importance: 0.81) in our model. Given that we are trying to outperform the spread by a large enough margin to profit on NBA bets, we needed to drop this feature from our training set and develop a different approach (additional model types, new features, and robust cross validation across multiple time horizons).
2. Run the final_model_dev.ipynb notebook, which generates a series of random forest, neural network, and XGBoost models. These model predictions are ultimately used as features in a meta-model using linear regression for a final prediction. 