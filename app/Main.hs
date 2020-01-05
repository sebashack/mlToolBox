module Main where

import Numeric.LinearAlgebra.Data (matrix, vector, (?), fromLists)

import Regression.Linear (computeCost, gradientDescent, featureNormalize)

main :: IO ()
main = do
  let examples = fromLists [ [100, 10000, 2 , 0.5,  455]
                           , [50 , 50000, 6 , 0.7,  800]
                           , [400, 90000, 10, 0.11, 812] ]
  print $ featureNormalize examples
