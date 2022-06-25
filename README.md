# Location-matching

This model matches using a custom keras neural network which takes in location and subcategory 

search space is reduced using KD tree KNN

uvicorn main:app

Docker build -f Dockerfile . 


Sample input --- send it as raw json

{

"anchor_id": "E_001b6bad66eb98",

"validate_id": "E_0283d9f61e569d",

"anchor": [-3.01467,104.794373,"Stadiums"],
"validate": [3.021727,104.788628,"Soccer Stadiums"]

}
