module Main where

import Numeric.LinearAlgebra.Data ((?), fromLists, matrix, vector)

import Regression.Common (featureNormalize)
import Regression.Linear (computeCost, gradientDescent)

main :: IO ()
main = do
  let examples =
        fromLists
          [ [100, 10000, 2, 0.5, 455]
          , [50, 50000, 6, 0.7, 800]
          , [400, 90000, 10, 0.11, 812]
          ]
  print $ featureNormalize examples
