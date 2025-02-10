"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$
def is_close(a:float,b:float):
    return abs(a-b) < 1e-2

# TODO: Implement for Task 0.1.

def add(a:float,b:float):
    return a+b
def mul(a:float,b:float):
    return a*b
def neg(a:float):
    return -a
def max(a:float,b:float):
    if a>b:
        return a
    return b
def inv(a:float):
    return 1/a
def id(a:float):
    return a
def lt(a:float,b:float):
    return a < b
def eq(a:float,b:float):
    return a == b
def sigmoid(a:float):
    if a >= 0:
        return (1.0/(1.0+math.exp(-a)))
    else:
        return (math.exp(a)/(1.0+math.exp(a)))
def relu(a:float):
    return max(0.0,a)
def log(a:float):
    return math.log(a)
def exp(a:float):
    return math.exp(a)
def log_back(a:float,b:float):
    return b/a
def inv_back(a:float,b:float):
    return -b/(a**2)
def relu_back(a:float,b:float):
    if a > 0:
        return b
    return 0
# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], xs: list[float]) -> list[float]:
    return [fn(x) for x in xs]
def zipWith(fn: Callable[[float, float], float], xs: list[float], ys: list[float]) -> list[float]:
    return [fn(x,y) for x,y in zip(xs,ys)]
def reduce(fn: Callable[[float, float], float], xs: list[float], start: float) -> float:
    res = start
    for x in xs:
        res = fn(res,x)
    return res
def negList(xs: list[float]) -> list[float]:
    return map(neg,xs)
def addLists(xs: list[float], ys: list[float]) -> list[float]:
    return zipWith(add,xs,ys)
def sum(xs: list[float]) -> float:
    return reduce(add,xs,0.0)
def prod(xs: list[float]) -> float:
    return reduce(mul,xs,1.0)