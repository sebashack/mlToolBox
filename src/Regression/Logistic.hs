module Regression.Logistic
  ( computeCost
  , gradientDescent
  ) where

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
import Regression.Common (sigmoid)

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

gradientDescent ::
     Matrix R -> Vector R -> Vector R -> R -> Int -> Maybe R -> Vector R
gradientDescent x y theta alpha depth maybeRegParam = go 0 theta
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
            accum' =
              if i == 0
                then accum
                else accum * regFactor
         in computeGd i (accum' - v)
      where
        m = size y
        --
        regFactor =
          maybe
            (scalar (1 :: R))
            (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
            maybeRegParam
