from sympy import *
import numpy as np

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Annotated
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool

class StoppingCriteria(BaseModel):
    max_iterations: int
    max_error: float
class BodyType(BaseModel):
    auto_differentiate: bool
    variables: List[str]
    eqns: List[Dict[str, str]]
    initial_values: Dict[str, float]
    stopping_criteria: StoppingCriteria
# class Result(BaseModel):
#             results: List[{
#                  iteration: int,
#             values: Dict[str, float],
#             errors: Dict[str, float]
#             }]

# @app.put("/systems-of-nonlinear-equations/newton-raphson")
# def update_item(autoDifferentiate: int, item: Parameters):
#     return {"item_name": item.name, "item_id": item_id}
    

@app.get("/")
def home():
    return {"msg": "Welcome in my api"}

@app.post("/systems-of-nonlinear-equations/newton-raphson")
async def jacobi_iteration(bodyValues: BodyType):
    try:
        if bodyValues.auto_differentiate:
            jacobi_array = [[diff(sympify(item["eqn"]), v) for v in bodyValues.variables] for item in bodyValues.eqns]
        else:
            jacobi_array = [[sympify(item[v]) for v in bodyValues.variables] for item in bodyValues.eqns]
        
        print(jacobi_array)

            
        f_array = [sympify(item["eqn"]) for item in bodyValues.eqns]
    except:
        raise HTTPException(status_code=404, detail="Invalid Equation")
        
    prev_values = np.array([bodyValues.initial_values[key] for key in bodyValues.variables])

    results = []

    i = 0
    while i < bodyValues.stopping_criteria.max_iterations:  # Perform 4 iterations (as per the original code)
        jacobian = np.array([[float(item.evalf(subs={key: prev_values[ind] for ind, key in enumerate(bodyValues.variables)})) for item in row] for row in jacobi_array])
        f = np.array([float(item.evalf(subs={key: prev_values[ind] for ind, key in enumerate(bodyValues.variables)}) * -1) for item in f_array]).T

        delta = np.linalg.inv(jacobian).dot(f)
        values = delta + prev_values

        relative_absolute_errors = {}
        for ind, var in enumerate(bodyValues.variables):
            if prev_values[ind] != 0:
                relative_absolute_errors[var] = abs(delta[ind] / prev_values[ind]) * 100
            else:
                relative_absolute_errors[var] = None

        prev_values = values.copy()

        result = {
            "iteration": i+1,
            "values": {key:values[ind] for ind, key in enumerate(bodyValues.variables)},
            "errors": relative_absolute_errors
        }
        results.append(result)
        print(result, i)
        i+=1

        c = [relative_absolute_errors[x] < bodyValues.stopping_criteria.max_error for x in bodyValues.variables]
        print (c)

        if all([relative_absolute_errors[x] < bodyValues.stopping_criteria.max_error for x in bodyValues.variables]):
            break

    return {"results": results}



# variables = ["x","y"]
# eqns = [{"eqn": "x**2 + x*y - 10", "x": "2*x + y", "y": "x"}, {"eqn": "y + 3*x*y**2 - 57", "x": "3*y**2", "y": "1 + 6*x*y"}]
# initialValue = {
#   "x": 1.5,
#   "y": 3.5
# }

# jacobiArray = [[sympify(item[v]) for v in variables] for item in eqns]
# fArray = [sympify(item["eqn"]) for item in eqns]
# prevValues = np.array([initialValue[key] for key in initialValue])


# for i in range(4):
#   jacobian = np.array([[float(item.evalf(subs={key: prevValues[ind] for ind, key in enumerate(variables)})) for item in row] for row in jacobiArray])
#   f = np.array([float(item.evalf(subs={key: prevValues[ind] for ind, key in enumerate(variables)}) * -1) for item in fArray]).T

#   delta = np.linalg.inv(jacobian).dot(f)
#   values = delta + prevValues

#   prevValues = values.copy()

#   print(i, ">>>>>>>>>")
#   print(jacobian)
#   print(f)
#   print(delta)
#   print(values)
#   print()
  



# for i in range(1):

