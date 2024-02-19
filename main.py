from sympy import *
import numpy as np

variables = ["x","y"]
eqns = [{"eqn": "x**2 + x*y - 10", "x": "2*x + y", "y": "x"}, {"eqn": "y + 3*x*y**2 - 57", "x": "3*y**2", "y": "1 + 6*x*y"}]
initialValue = {
  "x": 1.5,
  "y": 3.5
}

jacobiArray = [[sympify(item[v]) for v in variables] for item in eqns]
fArray = [sympify(item["eqn"]) for item in eqns]
prevValues = np.array([initialValue[key] for key in initialValue])


for i in range(4):
  jacobian = np.array([[float(item.evalf(subs={key: prevValues[ind] for ind, key in enumerate(variables)})) for item in row] for row in jacobiArray])
  f = np.array([float(item.evalf(subs={key: prevValues[ind] for ind, key in enumerate(variables)}) * -1) for item in fArray]).T

  delta = np.linalg.inv(jacobian).dot(f)
  values = delta + prevValues

  prevValues = values.copy()

  print(i, ">>>>>>>>>")
  print(jacobian)
  print(f)
  print(delta)
  print(values)
  print()
  



# for i in range(1):

