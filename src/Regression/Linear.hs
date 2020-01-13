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
  , tr'
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
    alphaDivM = scalar $ (alpha / fromIntegral m)
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
    computeDelta th = (tr' x) #> ((x #> th) - y)
    --
    regFactor =
      maybe
        (scalar (1 :: R))
        (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
        maybeRegParam
