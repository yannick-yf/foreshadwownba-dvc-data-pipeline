# NBA Games Prediction Pipeline

This repository contains a data pipeline for generating the datasets used for modeling and predicting the outcomes of NBA games. The pipeline is built using DVC (Data Version Control) and consists of the following stages:

## Stages

### 1. Get Training Dataset

This stage retrieves the necessary training datasets from a database or other sources. It downloads two datasets:

- `player_attributes_salaries_dataset.csv`: Contains player attributes and salary information.
- `nba_games_training_dataset.csv`: Contains historical NBA game data.

### 2. Pre-Cleaning Dataset

In this stage, the `nba_games_training_dataset.csv` is pre-cleaned by removing unnecessary data, such as games played during the COVID-19 bubble, and selecting relevant columns.

### 3. Feature Engineering Pipeline

This stage performs feature engineering on the pre-cleaned dataset. It applies various transformations and calculations to generate new features that may be useful for predicting game outcomes.

### 4. Get Y Variables

In this stage, the target variables (y variables) are generated from the feature-engineered dataset. These variables represent the quantities that the machine learning model will try to predict, such as the winning team or the probability of winning.

### 5. Load Training Dataset to Database

The final stage loads the processed training dataset, including the engineered features and target variables, into a database for further use in model training and evaluation.

## Getting Started

To run this pipeline locally, follow these steps:

1. Clone the repository:

git clone https://github.com/your-repo/nba-games-prediction-pipeline.git


2. Install the required dependencies:

pip install -r requirements.txt


3. Set up the necessary environment variables for database connections and other configurations.

4. Run the DVC pipeline:

dvc repro


This command will execute all the stages in the pipeline, from retrieving the training datasets to loading the final processed dataset into the database.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
