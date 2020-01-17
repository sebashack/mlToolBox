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
  , toList
  , tr'
  )

computeCost :: Matrix R -> Vector R -> Vector R -> R
computeCost x y theta =
  let m = fromIntegral $ size y
      a = (x #> theta) + (-1 * y)
   in (a <.> a) / (2 * m)

gradientDescent :: Matrix R -> Vector R -> Vector R -> R -> Int -> R -> Vector R
gradientDescent _ _ _ alpha numIters regFactor
  | regFactor < 0 = error "Regularization factor cannot be < 0"
  | alpha < 0 = error "Learning factor cannot be < 0"
  | numIters < 0 = error "Number of iterations cannot be < 0"
gradientDescent x y theta alpha numIters regFactor = go 0 theta
  where
    m = fromIntegral $ size y
    --
    alpha' = scalar alpha
    --
    go k accum
      | k >= numIters = accum
      | otherwise =
        let delta = computeDelta accum
         in go (k + 1) (accum - (alpha' * delta))
    --
    penalizedTheta = fromList $ 0 : (tail $ ((* regFactor) <$> toList theta))
    --
    computeDelta th =
      let delta = (tr' x) #> ((x #> th) - y)
       in (delta + penalizedTheta) / scalar m
