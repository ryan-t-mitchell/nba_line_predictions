from pydantic import BaseModel, validator, ValidationError
from fastapi import FastAPI, HTTPException
from datetime import datetime as dt
import numpy as np
import joblib

# Load model
model = joblib.load("./random_forest_model.pkl")

# Create pydantic models for model data input/output validations
class Input(BaseModel):
    home_team_days_rest: int
    home_team_home_prior: int
    home_team_sos: float
    home_team_sos_last_10: float
    home_team_win_pct: float
    home_team_win_pct_last_10: float
    home_team_3pt_pct: float
    home_team_2pt_pct: float
    home_team_pp100p: float
    home_team_orb_pct: float
    home_team_drb_pct: float
    home_team_opp_3pt_pct: float
    home_team_opp_2pt_pct: float
    home_team_opp_pp100p: float
    
    away_team_days_rest: int
    away_team_home_prior: int
    away_team_sos: float
    away_team_sos_last_10: float
    away_team_win_pct: float
    away_team_win_pct_last_10: float
    away_team_3pt_pct: float
    away_team_2pt_pct: float
    away_team_pp100p: float
    away_team_orb_pct: float
    away_team_drb_pct: float
    away_team_opp_3pt_pct: float
    away_team_opp_2pt_pct: float
    away_team_opp_pp100p: float

# Pydantic data model to validate LIST of Input vectors (games)
class Input_List(BaseModel):
    games: list[Input]

class Output(BaseModel):
    home_team_plus_minus: float

    # @validator('MedHouseValue')
    # def medhousevalue(cls,v):
    #     assert isinstance(v,float), 'Output value must be a float'
    #     return v

# Pydantic data model to validate LIST of Outputs
class Output_List(BaseModel):
    home_team_plus_minus_predictions: list[Output]

# Start FastAPI
app = FastAPI()

#@app.get("/") Root is not included because framework will handle this missing endpoint with a 404 error automatically. No need to code it!

@app.get("/hello")
async def read_name(name: str):
    return "hello {}".format(name)

@app.post("/predict", response_model = Output_List) # Response from post will be an "Output" data model type that we created
# Declare a parameter 'data' and its type will 'Input' from our data model
# FastAPI will read the body of the request as JSON.
async def predict(data: Input_List):
    for i, game in enumerate(data.games):
    # Convert data (type = Input_List data model) into a dictionary, then get the values and dump into a list
        data_dict = game.dict()
        data_list = list(data_dict.values())
        tmp_array = np.array(data_list)
        # Reshape list for inputting into our model
        tmp_array = tmp_array.reshape(-1,len(data_dict))
        if i == 0:
            # Instantiate our data array
            data_array = tmp_array
        else:
            # Stack onto our existing data array
            data_array = np.vstack([data_array, tmp_array])

    # predictions = list(model.predict(data_array))
    # return {"home_team_plus_minus_predictions": predictions}
    predictions = model.predict(data_array)
    # Convert predictions to a list of dictionaries
    predictions_list = [{"home_team_plus_minus": pred} for pred in predictions]
    return {"home_team_plus_minus_predictions": predictions_list}



@app.get("/health")
async def datetime():
    return dt.now()