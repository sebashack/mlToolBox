module Main where

import Numeric.LinearAlgebra.Data (matrix, vector, (?))

import Regression.Linear (computeCostFunction, gradientDescent)

main :: IO ()
main = do
  let examples = matrix 5 [ 1, 1, 2, 3, 4
                          , 1, 5, 6, 7, 8
                          , 1, 9, 10, 11, 12]
      r = computeCostFunction examples (vector [20, 21, 22]) (vector [0, 0, 0, 0, 0])
      solutions = gradientDescent examples (vector [20, 21, 22]) (vector [0, 0, 0, 0, 0]) 0.01 2000
  print solutions
