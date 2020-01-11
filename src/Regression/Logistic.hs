module Regression.Logistic
  ( computeCost
  , gradientDescent
  ) where

import Numeric.LinearAlgebra (( #> ), (<.>), add)
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
  , tr'
  )
import Regression.Common (sigmoid, sigmoidVec)

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
         in go (i + 1) (accum + a - b)
      where
        m = size y

gradientDescent ::
     Matrix R -> Vector R -> Vector R -> R -> Int -> Maybe R -> Vector R
gradientDescent x y theta alpha depth maybeRegParam = go 0 theta
    --
  where
    alphaDivM = scalar $ (alpha / fromIntegral m)
    --
    m = size y
    --
    go k accum
      | k >= depth = accum
      | otherwise =
        let delta = computeDelta accum
            accum' =
              if k == 0
                then accum
                else accum * regFactor
         in go (k + 1) (accum' - (alphaDivM * delta))
    --
    xTr = tr' x
    --
    computeDelta th = xTr #> ((sigmoidVec (x #> th)) - y)
    --
    regFactor =
      maybe
        (scalar (1 :: R))
        (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
        maybeRegParam
