# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 23:56:31 2022

@author: abhin
"""


from fastapi import FastAPI,Body

app = FastAPI()

from src.utils import predict,build_model
model = build_model()
model.load_weights('src/weights/cosine.h5')



@app.get("/health_check")
async def root():
    return {"message": "service is up and running"}


@app.post("/predict")
async def predict_point(anchor_validate_dict:dict=Body("")):
    print(anchor_validate_dict)
    
    anchor_id = anchor_validate_dict['anchor_id']
    validate_id = anchor_validate_dict['validate_id']
    
    anchor = anchor_validate_dict['anchor']
    validate = anchor_validate_dict['validate']
    
    pred = predict(model, anchor, validate)
    
    if pred:
        return "{} {} points to same point of inetrest".format(anchor_id,validate_id)
    else:
        return "{} {} does not point to same point of inetrest".format(anchor_id,validate_id)
