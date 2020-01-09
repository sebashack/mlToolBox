module Regression.Logistic where

import Numeric.LinearAlgebra ((<.>))
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , (?)
  , atIndex
  , cols
  , flatten
  , fromList
  , scalar
  , size
  )

sigmoid :: R -> R
sigmoid z = 1 / (1 + (e ** (-z)))
  where
    e = exp 1

computeCost :: Matrix R -> Vector R -> Vector R -> R
computeCost x y theta = go 0 0
  where
    go i accum
      | i >= m = accum / (fromIntegral m)
      | otherwise =
        let xVals = flatten $ x ? [i]
            s = sigmoid (theta <.> xVals)
            yi = y `atIndex` i
            a = (-1 * yi) * (log s)
            b = (1 - yi) * (log $ 1 - s)
         in go i (accum + a - b)
      where
        m = size y

gradientDescent :: Matrix R -> Vector R -> Vector R -> R -> Int -> Vector R
gradientDescent x y theta alpha depth = undefined
  where
    go k th
      | k >= depth = th
      | otherwise =
        let gd = computeGd 0 (fromList $ replicate (cols x) 0)
         in go (k + 1) (th - gd)
    --
    alpha' = scalar alpha
    --
    computeGd i accum
      | i >= m = accum
      | otherwise =
        let xVals = flatten $ x ? [i]
            s = sigmoid (theta <.> xVals)
            v =
              (alpha' / fromIntegral m) *
              ((scalar (s - (y `atIndex` i))) * xVals)
         in computeGd i (accum - v)
      where
        m = size y
