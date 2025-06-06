from typing import List

## definition of a vector as a list of floats.
## A vector is a mathematical object that has both a magnitude and a direction.
## In Python, a vector can be represented as a list of floats.

Vector = List[float]

## The dot product of two vectors is a scalar value that is the sum of the products of their 
## corresponding components.
## The dot product is a way to multiply two vectors together to get a single number.
## The dot product is also known as the scalar product or inner product.

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

## Suppose I have a function `f` that takes as input a vector of real numbers 
## and outputs a single real number.
## definition of the function `f`. regular standalone function, not tied to a class or object.
## function named sum_of_squares that takes a vector v as input and returns a float.
## dot(v,v) is used instead of a loop to compute the sum of squares (loop to square each element 
## and sum them up). 
## The dot product is a common operation in linear algebra, and it can be used to compute the
## sum of squares of a vector by taking the dot product of the vector with itself.

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)

## Frequently, I need to find the input v that minimizes or maximizes the value of `f(v)`.

## To maximize or minimize the function `f`, I can use a gradient ascent or descent algorithm.
## To maximize a function is to pick a random start point, compute the gradient at that point,
## and then take a step in the direction of the gradient.
## To minimize a function is to pick a random start point, compute the gradient at that point,
## and then take a step in the opposite direction of the gradient.

## the next function approximates the derivative of `f` at a point `x` using a small number h.
## h is a small number that represents the step size. (simulate an infinitesimal change in x).

from typing import Callable ## means sth. I can pass a function as an argument to another function.

## for the next function firstly define the type of `f` as a callable that takes a float and returns a float.
## x is the point at which I want to compute the derivative, and h is a small number that represents
## the step size. The function returns the difference quotient, which is an approximation of the derivative.
## difference quotient is the ratio of the change in the function value to the change in the input value.
## in other words, h is the slope between two points very close to each other on the function f.

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h ## approximation of the derivative of f at x.

## Note when f is a function of many variables, it has multiple partial derivatives.
## The partial derivative of f with respect to the i-th variable is the derivative of f
## with respect to that variable, holding all other variables constant.

## APPROXIMATION OF THE PARTIAL DERIVATIVE OF A MULTIVARIATE FUNCTION:
def partial_difference_quotient(f: Callable[[Vector], float],
                                 v: Vector,
                                 i: int,
                                 h: float) -> float:
    """Compute the i-th partial difference quotient of f at v"""
    w = v[:]
    w[i] += h
    return (f(w) - f(v)) / h  ## approximation of the partial derivative of f at v with respect to the i-th variable.

## After that, I can compute the gradient of `f` at a point `v` in the same way.
## TO do it I need to loop over each component i of the vector `v` and 
## compute the partial derivative of `f` with respect to that component.
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.00001) -> Vector:
    """Estimate the gradient of f at v"""
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]  ## list comprehension to compute the gradient.




## CHOOSING A STEP SIZE:

## The step size is a small number that determines how far to move in the direction of the gradient.
## If the step size is too small, the algorithm will take a long time to converge.
## If the step size is too large, the algorithm may overshoot the minimum and diverge.
## A common approach is to start with a small step size and increase it gradually until the algorithm
## converges but it is costly to compute the gradient at each step.

## FOr now I will use a fixed step size.

## Here gradient descent will be use to fit parameterized models to data.
## Usual case is to have some dataset and some model that predicts the output from the input.

## Loss function will be used to measure how well the model fits the data.(smaller is better).
## This means I can use gradient descent to minimize the loss function 
## and find the best parameters for the model.

## Example:

    # x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad

## Interpretation of the function `linear_gradient`:



