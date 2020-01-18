{-# LANGUAGE RecordWildCards #-}

module Regression.Linear
  ( computeCost
  , gradientDescent
  , minimizeBFGS2
  ) where

import Numeric.GSL.Minimization (MinimizeMethodD(VectorBFGS2), minimizeVD)
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
import Regression.Common (MinimizationOpts(..))

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

minimizeBFGS2 ::
     Matrix R
  -> Vector R
  -> Vector R
  -> Int
  -> MinimizationOpts
  -> R
  -> Vector R
minimizeBFGS2 _ _ _ numIters _ regFactor
  | regFactor < 0 = error "Regularization factor cannot be < 0"
  | numIters < 0 = error "Number of iterations cannot be < 0"
minimizeBFGS2 x y theta numIters MinimizationOpts {..} regFactor =
  let (params, _) =
        minimizeVD
          VectorBFGS2
          precision
          numIters
          sizeOfFirstTrialStep
          tolerance
          (computeCost x y)
          computeGradients
          theta
   in params
  where
    m = fromIntegral $ size y
    --
    xTr = tr' x
    --
    penalizedTheta = fromList $ 0 : (tail $ ((* regFactor) <$> toList theta))
    --
    computeGradients :: Vector R -> Vector R
    computeGradients th =
      let delta = (tr' x) #> ((x #> th) - y)
       in (delta + penalizedTheta) / scalar m
