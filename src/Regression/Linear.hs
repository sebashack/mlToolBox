module Regression.Linear
  ( computeCost
  , gradientDescent
  ) where

import Numeric.LinearAlgebra (( #> ), (<.>), (?))
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , atIndex
  , cols
  , flatten
  , fromList
  , scalar
  , size
  )

computeCost :: Matrix R -> Vector R -> Vector R -> R
computeCost x y theta =
  let m = fromIntegral $ size y
      a = (x #> theta) + (-1 * y)
   in (a <.> a) / (2 * m)

gradientDescent ::
     Matrix R -> Vector R -> Vector R -> R -> Int -> Maybe R -> Vector R
gradientDescent x y theta alpha depth maybeRegParam = go 0 theta
  where
    m = size y
    --
    alpha' = scalar alpha
    --
    zerosDelta = fromList $ replicate (cols x) 0
    --
    go k accum
      | k >= depth = accum
      | otherwise =
        let delta = computeDelta accum 0 zerosDelta
            accum' =
              if k == 0
                then accum
                else accum * regFactor
         in go (k + 1) (accum' - (alpha' * delta))
    --
    computeDelta :: Vector R -> Int -> Vector R -> Vector R
    computeDelta th i delta
      | i >= m = delta / (scalar $ fromIntegral m)
      | otherwise =
        let xVals = flatten $ x ? [i]
            delta' =
              delta + ((scalar $ (th <.> xVals) - (y `atIndex` i)) * xVals)
         in computeDelta th (i + 1) delta'
    --
    regFactor =
      maybe
        (scalar (1 :: R))
        (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
        maybeRegParam
