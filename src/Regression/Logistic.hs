{-# LANGUAGE RecordWildCards #-}

module Regression.Logistic
  ( computeCost
  , gradientDescent
  , gradientBFGS2
  ) where

import Numeric.GSL.Minimization (MinimizeMethodD(VectorBFGS2), minimizeVD)
import Numeric.LinearAlgebra (( #> ), (<.>), add)
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , (?)
  , atIndex
  , cmap
  , cols
  , flatten
  , fromList
  , scalar
  , size
  , toList
  , tr'
  )
import Regression.Common (MinimizationOpts(..), sigmoid, sigmoidVec)

computeCost :: Matrix R -> Vector R -> Vector R -> R
computeCost x y theta =
  let s = sigmoidVec (x #> theta)
      m = size y
   in ((y <.> log (s)) + ((1 - y) <.> log (1 - s))) / fromIntegral (-m)

gradientDescent :: Matrix R -> Vector R -> Vector R -> R -> Int -> R -> Vector R
gradientDescent _ _ _ alpha numIters regFactor
  | regFactor < 0 = error "Regularization factor cannot be < 0"
  | alpha < 0 = error "Learning factor cannot be < 0"
  | numIters < 0 = error "Number of iterations cannot be < 0"
gradientDescent x y theta alpha depth regFactor = go 0 theta
    --
  where
    alpha' = scalar alpha
    --
    m = fromIntegral $ size y
    --
    go k accum
      | k >= depth = accum
      | otherwise =
        let delta = computeDelta accum
         in go (k + 1) (accum - (alpha' * delta))
    --
    penalizedTheta = fromList $ 0 : (tail $ ((* regFactor) <$> toList theta))
    --
    computeDelta th =
      let delta = (tr' x) #> ((sigmoidVec (x #> th)) - y)
       in (delta + penalizedTheta) / scalar m

gradientBFGS2 ::
     Matrix R
  -> Vector R
  -> Vector R
  -> Int
  -> MinimizationOpts
  -> R
  -> Vector R
gradientBFGS2 _ _ _ numIters _ regFactor
  | regFactor < 0 = error "Regularization factor cannot be < 0"
  | numIters < 0 = error "Number of iterations cannot be < 0"
gradientBFGS2 x y theta numIters MinimizationOpts {..} regFactor =
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
      let delta = (tr' x) #> ((sigmoidVec (x #> th)) - y)
       in (delta + penalizedTheta) / scalar m
