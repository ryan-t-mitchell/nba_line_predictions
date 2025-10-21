# Project Airball 

<div align="center">
<img max-height='200px' src='project_airball/assets/air-ball.png'>
</div>

<img max-height='400px' src='project_airball/assets/dailypicks.png'>

## Website
https://air-ball.vercel.app

## Details
The goal of this project is to make a profitable betting algorithm using NBA regular season data and inital point spreads across a series of betting sites. This repo covers the ML algorithm development and hosting of the final model in AWS. 

In order to bring the project to life, I have collaborated with a friend who built the [front end and back end](https://github.com/birdman093/air-ball) which calls the NBA API to pull in daily box-score data, calculates updated feature values, and then calls my algorithm for predictions using the updated feature vectors for the next day's games. The application also tracks betting performance during the NBA season. 

Give it a try! Running this CURL command will give you the predicted game betting line for a game:

```curl -X POST -H 'Content-Type: application/json' http://ec2-35-172-137-50.compute-1.amazonaws.com:8000/predict -d '{"games": [{"home_team_days_rest": 1.0, "home_team_home_prior": 0.0, "home_team_sos": 14.901234567901234, "home_team_sos_last_10": 17.2, "home_team_win_pct": 0.5308641975308642, "home_team_win_pct_last_10": 0.5, "home_team_3pt_pct": 0.3425925925925926, "home_team_2pt_pct": 0.5385365853658537, "home_team_pp100p": 109.57242744879647, "home_team_orb_pct": 0.21107544141252, "home_team_drb_pct": 0.6857142857142857, "home_team_opp_3pt_pct": 0.3672903672903673, "home_team_opp_2pt_pct": 0.5694227769110765, "home_team_opp_pp100p": 109.99381570810142, "away_team_days_rest": 1.0, "away_team_home_prior": 0.0, "away_team_sos": 14.728395061728396, "away_team_sos_last_10": 13.7, "away_team_win_pct": 0.419753086419753, "away_team_win_pct_last_10": 0.5, "away_team_3pt_pct": 0.3458466453674121, "away_team_2pt_pct": 0.5399553571428571, "away_team_pp100p": 108.21244455101306, "away_team_orb_pct": 0.2216815355501487, "away_team_drb_pct": 0.7177635098983414, "away_team_opp_3pt_pct": 0.3503014065639652, "away_team_opp_2pt_pct": 0.5662888122227698, "away_team_opp_pp100p": 110.53451581975072}]}'```

Historical odds data downloaded from: https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/tree/master/Odds-Data/Odds-Data-Clean

## Data Gathering Steps
1. First, run the final_team_stats.ipynb file, which pings the NBA API for relevant box score data which will feed into our final feature set.
2. Next, download historical odds data from the repo linked above (2015-16 through 2021-22 seasons).
3. Run the odds.ipynb file, which concatenates historical odds data into a single dataframe for processing. For 2022-23 and later seasons, scrape the sports betting sites and append the mode of the home team spread values.
4. Construct the training data set using the NBA API data and the odds data (construct_training_set.ipynb).

## Modeling
1. Our initial model (created in initial_model.ipynb) generates a random forest machine learning model using the HOME SPREAD from the betting sites, along with custom features that I engineered. This initial model failed to outperform a model using just the HOME SPREAD. Perhaps unsurprisingly, the HOME SPREAD feature was by far the most important feature (feature importance: 0.81) in our model. Given that we are trying to outperform the spread by a large enough margin to profit on NBA bets, we needed to drop this feature from our training set and develop a different approach (additional model types, new features, and robust cross validation across multiple time horizons).
2. Run the final_model_dev.ipynb notebook, which generates a series of random forest, neural network, and XGBoost models. These model predictions are ultimately used as features in a meta-model using linear regression for a final prediction.

## Build Process
poetry new project --name src
cd project/src
poetry add uvicorn[standard]
poetry add nba_api
poetry add requests
poetry add ordered_set
poetry add fastapi
poetry add pytest
poetry add sbrscrape
poetry add openpyxl
poetry add tensorflow
poetry add scipy
poetry add scikit-learn
poetry add keras-tuner
poetry add deepctr

# Docker build (from the /project folder)
docker build -t nba:1.0 .
docker run --rm --name nba_project -p 8000:8000 -d nba:1.0

# Push image to Docker Hub
docker login --username=rmitchell88
docker tag nba:1.0 rmitchell88/nba:1.0
docker push rmitchell88/nba:1.0

# SSH into EC2 instance
cd Downloads/
ssh -i "w251-keypair.pem" ubuntu@ec2-35-172-137-50.compute-1.amazonaws.com #ec2-54-84-37-130.compute-1.amazonaws.com
sudo su
apt-get update
apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
    
# add the docker repo    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
 
# install Docker & Vim
sudo apt update
apt-get update
apt-get install docker-ce
apt-get install -y vim
apt install python3-dev python3-pip
sudo apt install python3-pip python3-venv -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install
<!-- pip3 install numpy
pip3 install Cython
apt install awscli -->

# Pull the image onto EC2 instance (make sure port 8000 is open on the instance, which is the default listening port for uvicorn)
docker pull rmitchell88/nba:1.0
docker run --rm --name nba_project -p 8000:8000 -d rmitchell88/nba:1.0