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


from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, Ridge,
                                  RidgeCV)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import scale

# ## Regression



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

# ## Appearances Dataset: Exploration and Tranformation

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


# #### Further Feature Engineering From Corr. Matrix

players = players.drop(columns=['height_in_cm', 'year'])


# ## Encoding values

players = pd.get_dummies(players)

players = players.sort_values(by=['player_id'])


# #### Preparing for Regression

X = players.drop(columns=['market_value_in_eur', 'player_id'])
y = players['market_value_in_eur']

# Given the performance results from the notebook, Lasso Regression will be used to carry out the prediction.
# ## Lasso Regression

lasso_reg = Lasso().fit(X, y)

new_player = pd.read_csv('new_player.csv')

# #### Test Set Prediction

y_predicted = lasso_reg.predict(new_player)
print("Predicted Market Value: ", y_predicted[0].round())
