#!/usr/bin/env python
# coding: utf-8

# # Final Project Group 6
# Group members:
# - Niccolo Matteo Borgato
# - Nicholas Dieke
# - Austin Brees
# - Sophie Schaesberg
# - Adrian Marino

# ## Introduction
# 
# Football is one of the most popular sports in the world, with millions of fans and billions of dollars in revenue generated each year. One of the key factors that contributes to the success of a football team is the quality of its players, and as a result, the market value of football players has become a crucial metric for teams, agents, and fans alike. In this machine learning problem, we aim to predict the market value of football players using a dataset of various player attributes such as age, position, and performance statistics. By accurately predicting a player's market value, we can provide valuable insights to clubs and agents on potential player transfers, and help fans to better understand the true value of their favorite players. This is an exciting and challenging problem that requires a combination of data analysis, statistical modeling, and machine learning techniques, and has the potential to revolutionize the way football clubs operate in the transfer market.

# ## Importing libraries

428


from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ## Importing Data

players = pd.read_csv('players.csv')
player_valuations = pd.read_csv('player_valuations.csv')
appearances = pd.read_csv('appearances.csv')
clubs = pd.read_csv('clubs.csv')


# ## Market Value over the years

# add year to player valuations
player_valuations['datetime']=pd.to_datetime(player_valuations['datetime'], format="%Y-%m-%d")
player_valuations['year']=player_valuations['datetime'].dt.year

#filter range
player_valuations_df = player_valuations[(player_valuations.year > 2004 ) & (player_valuations.year < 2023 )]
high_value_player_valuations_df = player_valuations_df[(player_valuations_df.market_value_in_eur > 40000000 )]
positions=players['position'].unique()

# Visualize of valuation data over time
plt.figure(figsize=(20,8))
plt.scatter(player_valuations_df['datetime'],y=player_valuations_df['market_value_in_eur']/1000000, c='red',alpha=0.15)
plt.xlabel('date');plt.ylabel('Valuation in million euros')
plt.title('Player valuations over time',fontsize=28)
plt.show()


# #### Dropping irrelevant features
# market_value_in_eur is here too because we will use the value from player_valuations.csv


players = players.drop(columns=['name', 'country_of_citizenship', 'sub_position', 'market_value_in_eur', 'current_club_name', 'country_of_birth', 'highest_market_value_in_eur', 'agent_name', 'city_of_birth', 'first_name', 'last_name', 'player_code', 'image_url', 'url'])


# #### Formatting dates


players['date_of_birth'] = pd.to_datetime(players['date_of_birth'])
players['contract_expiration_date'] = pd.to_datetime(players['contract_expiration_date'])
players['contract_expiration_year'] = players['contract_expiration_date'].dt.year
players = players.drop(columns=['contract_expiration_date'])


# #### Dropping irrelevant features

player_valuations = player_valuations.drop(columns=['date', 'dateweek', 'current_club_id', 'player_club_domestic_competition_id'])


# #### Formatting dates


player_valuations['datetime'] = pd.to_datetime(player_valuations['datetime'])
player_valuations['year'] = player_valuations['datetime'].dt.year
player_valuations = player_valuations.drop(columns=['datetime'])


# #### Averaging the market values per player per year

player_mean_year_val = player_valuations.groupby(['player_id', 'year'])['market_value_in_eur'].mean().reset_index()


# #### Dropping irrelevant features

appearances = appearances.drop(columns=['player_name', 'player_current_club_id', 'appearance_id'])


# #### Formatting dates

appearances['date'] = pd.to_datetime(appearances['date'])
appearances['year'] = appearances['date'].dt.year


# #### Grouped stats by player

players_stats_df = appearances.drop(columns=["game_id", "player_club_id"]).groupby(["player_id", "year"]).sum()
players_stats_df = players_stats_df.reset_index()


# ## Merging Datasets

players = players.merge(players_stats_df, on="player_id", how='inner').merge(player_mean_year_val, on=['player_id', 'year'], how='inner').merge(clubs[['club_id','stadium_seats', 'national_team_players']], left_on=['current_club_id'], right_on=['club_id'], how='inner')
players = players.drop(columns=['current_club_id', 'club_id'])


# #### Adding Age Feature

def calculate_age(dob, year):
    return year - dob.year

# drop players with no date of birth
players = players[players['date_of_birth'].isnull() == False]
today = date.today()
players['age'] = players.apply(lambda x: calculate_age(x.date_of_birth, x.year), axis=1)

players = players.drop(columns=['date_of_birth'])


# #### Dealing with null values

players['contract_expiration_year'] = players['contract_expiration_year'].fillna(players['last_season']+1)
players = players.drop(columns=['last_season'])
players['foot'] = players['foot'].fillna('Right')


# ## Correlation Matrix

# calculate the correlation matrix
corr_matrix = players.corr()

# create a heatmap with the correlation matrix
sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', center=0, ax=ax)

# set the plot title and axis labels
ax.set_title('Correlation Matrix of Merged Dataset', fontsize=16)
ax.set_xlabel('Features', fontsize=14)
ax.set_ylabel('Features', fontsize=14)

# show the plot
plt.show()


# #### Further Feature Engineering From Corr. Matrix

players = players.drop(columns=['height_in_cm', 'year'])


# ## Encoding values

players = pd.get_dummies(players)

players = players.sort_values(by=['player_id'])


# ---

# ## Regression

from sklearn.decomposition import PCA
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, Ridge,
                                  RidgeCV)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import scale

# #### Preparing for Regression

X = players.drop(columns=['market_value_in_eur', 'player_id'])
y = players['market_value_in_eur']

# Define cross-validation folds
cv = KFold(n_splits=10, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

# Run standardization on X variables
X_train = scale(X_train)
X_test = scale(X_test)


# ## Linear Regression

lin_reg = LinearRegression().fit(X_train, y_train)


# #### Get R2 score

lin_reg.score(X_train, y_train)


# #### Cross validation

lr_scores = abs(cross_val_score(lin_reg, 
                                 X_train, 
                                 y_train, 
                                 cv=cv, 
                                 scoring='neg_root_mean_squared_error'))


# #### Training Set Score

lr_score_train = np.mean(lr_scores)


# #### Test Set Prediction Score

y_predicted = lin_reg.predict(X_test)
lr_score_test = mean_squared_error(y_test, y_predicted, squared=False)


# ## Lasso Regression

lasso_reg = LassoCV().fit(X_train, y_train)


# #### Get R2 Score

lasso_reg.score(X_train, y_train)


# #### Cross validation

lasso_scores = abs(cross_val_score(lasso_reg, 
                                    X_train, 
                                    y_train, 
                                    cv=cv, 
                                    scoring='neg_root_mean_squared_error'))


# #### Training Set Score

lasso_score_train = np.mean(lasso_scores)


# #### Test Set Prediction Score

y_predicted = lasso_reg.predict(X_test)
lasso_score_test = mean_squared_error(y_test, y_predicted, squared=False)

# ## Ridge Regression (L2 regularization)

ridge_reg = RidgeCV().fit(X_train, y_train)


# #### Get R2 score

ridge_reg.score(X_train, y_train)


# #### Cross validation

ridge_scores = abs(cross_val_score(ridge_reg, 
                                    X_train, 
                                    y_train, 
                                    cv=cv, 
                                    scoring='neg_root_mean_squared_error'))


# #### Training Set Score

ridge_score_train = np.mean(ridge_scores)


# #### Test Set Prediction Score

y_predicted = ridge_reg.predict(X_test)
ridge_score_test = mean_squared_error(y_test, y_predicted, squared=False)


# ## Principal Components Regression

lin_reg = LinearRegression()
rmse_list = []


# ### First generate all the principal components

pca = PCA()
X_train_pc = pca.fit_transform(X_train)


# ### Loop through different count of principal components for linear regression

for i in range(1, X.shape[1]+1):
    rmse_score = abs(cross_val_score(lin_reg, 
                                      X_train_pc[:,:i], # Use first k principal components
                                      y_train, 
                                      cv=cv, 
                                      scoring='neg_root_mean_squared_error').mean())
    rmse_list.append(rmse_score)


# #### Plot RMSE per PC count

# Create a larger figure
fig = plt.figure(figsize=(16, 8))

# Plot the data
plt.plot(rmse_list, '-o')

# Add labels and title
plt.xlabel('Number of principal components in regression')
plt.ylabel('Cross-Validation RMSE')
plt.title('Transfer Value')

# Set the x-axis limits and tick labels
plt.xlim(xmin=-1)
plt.xticks(np.arange(X.shape[1]), np.arange(1, X.shape[1]+1))
plt.axhline(y=lr_score_train, color='g', linestyle='-');

# Show the plot
plt.show()


# #### Visually determine optimal number of principal components
# 

best_pc_num = 27


# #### Train model on training set

lin_reg_pc = LinearRegression().fit(X_train_pc[:,:best_pc_num], y_train)


# #### Get R2 score
# 

lin_reg_pc.score(X_train_pc[:,:best_pc_num], y_train)


# #### Cross validation

pcr_score_train = abs(cross_val_score(lin_reg_pc, 
                                       X_train_pc[:,:best_pc_num], 
                                       y_train, 
                                       cv=cv, 
                                       scoring='neg_root_mean_squared_error').mean())


# #### Get principal components of test set
# 

X_test_pc = pca.transform(X_test)[:,:best_pc_num]


# #### Predict on test data
# 

preds = lin_reg_pc.predict(X_test_pc)
pcr_score_test = mean_squared_error(y_test, preds, squared=False)


# ## Evaluation

train_metrics = np.array([round(lr_score_train,2), 
                          round(lasso_score_train,2), 
                          round(ridge_score_train,2), 
                          round(pcr_score_train,2)]) 
train_metrics = pd.DataFrame(train_metrics, columns=['RMSE (Train Set)'])
train_metrics.index = ['Linear Regression', 
                       'Lasso Regression', 
                       'Ridge Regression', 
                       f'PCR ({best_pc_num} components)']
print(train_metrics)



test_metrics = np.array([round(lr_score_test,2), 
                         round(lasso_score_test,2), 
                         round(ridge_score_test,2), 
                         round(pcr_score_test,2)]) 
test_metrics = pd.DataFrame(test_metrics, columns=['RMSE (Test Set)'])
test_metrics.index = ['Linear Regression', 
                      'Lasso Regression', 
                      'Ridge Regression', 
                      f'PCR ({best_pc_num} components)']
print(test_metrics)