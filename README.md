# Foreshadow DVC Data Pipeline

This repository contains a data pipeline that generates the training datasets to predict the various outcomes of NBA games. The pipeline is built using DVC package and consists of the following stages:

## Stages

### 1. Get Training Dataset

This stage retrieves the necessary datasets from a database or other sources. It downloads two datasets:

- `player_attributes_salaries_dataset.csv`: Contains player attributes and salary information.
- `nba_games_training_dataset.csv`: Contains data on historical NBA game logs.

These two datasets are generated from the data engineering pipeline performing an ETL process.
For more details, please refers to the:
- https://github.com/yannick-yf/foreshadwownba-data-engineering-pipeline

### 2. Pre-Cleaning Dataset

In this stage, the `nba_games_training_dataset.csv` is pre-cleaned by removing unnecessary data, such as games played during the COVID-19 bubble, and selecting relevant columns.

### 3. Feature Engineering Pipeline

The feature engineering pipeline is implemented in the `features_engineering.py` module. This module performs various feature engineering operations on the NBA games training dataset to create new features that can potentially improve the performance of the machine learning model.

The `features_engineering_pipeline` function is the main entry point for the feature engineering process. It reads the pre-cleaned training dataset and applies the following feature engineering steps:

- **Previous Games Average Features**: Calculates the average of various statistics (e.g., points scored, rebounds, assists) for each team based on their previous games.

- **Previous Games Ratio Average Features**: Calculates the ratio of various statistics for each team based on their previous games.

- **Previous Season Ratio Features**: Calculates the ratio of various statistics for each team based on their performance in the previous season.

- **Rest Days Between Games**: Calculates the number of rest days between consecutive games for each team.

- **Duration Trip Hours Between Cities**: Calculates the travel duration in hours between the cities of the two teams playing a game.

- **Previous Days Average Features**: Calculates the average of various statistics for each team based on their performance in the previous few days.

- **Extract Days of Week from Date**: Extracts the day of the week from the game date.

- **Game on Weekend Features**: Creates a binary feature indicating whether the game is played on a weekend or not.

- **Average Ratio Win/Lose away Game**: Calculates the average ratio of wins and losses for each team in away games.

- **Last Game Overtime**: Creates a binary feature indicating whether the last game for each team went into overtime or not.

- **Calculate Streak Features**: Calculates the current winning/losing streak for each team.

- **Final Cleaning**: Performs final cleaning or preprocessing steps on the engineered features. It removes the oldest season contains in the gamelogs dataset.

After applying all the feature engineering steps, the resulting dataset with the engineered features is saved as `nba_games_training_dataset_cleaned_w_features.csv` in the `data/processed` directory.

The `features_engineering_pipeline` function can be executed from the command line by providing the path to the configuration file as an argument (`--config-params`).

### 4. Get Y Variables

In this stage, the target variables are generated, such as the winning team.

### 5. Load Training Dataset to Database

The final stage loads the processed training dataset, including the engineered features and target variables, into a database for further use in model training and evaluation.

## Getting Started

To run this pipeline locally, follow these steps:

1. Clone the repository:

git clone https://github.com/your-repo/nba-games-prediction-pipeline.git

2. Install the required dependencies:

poetry shell

poetry update

3. Set up the necessary environment variables for database connections and other configurations.

MYSQL_USERNAME=

MYSQL_PASSWORD=

MYSQL_DATABASE=

MYSQL_HOST=

4. Run the end-to-end data pipeline:

dvc repro


This command will execute all the stages in the pipeline, from retrieving the training datasets to loading the final processed dataset into the database.

## Contributing


Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
