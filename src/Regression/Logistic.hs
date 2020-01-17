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
    computeDelta th = (tr' x) #> ((sigmoidVec (x #> th)) - y)
    --
    regFactor =
      maybe
        (scalar (1 :: R))
        (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
        maybeRegParam

gradientBFGS2 ::
     Matrix R
  -> Vector R
  -> Vector R
  -> Int
  -> MinimizationOpts
  -> Maybe R
  -> Vector R
gradientBFGS2 x y theta numIter MinimizationOpts {..} maybeRegParam =
  let (params, _) =
        minimizeVD
          VectorBFGS2
          precision
          numIter
          sizeOfFirstTrialStep
          tolerance
          (computeCost x y)
          computeGradients
          theta
   in params
  where
    m = size y
    --
    xTr = tr' x
    --
    computeGradients :: Vector R -> Vector R
    computeGradients th =
      let newTh = xTr #> ((sigmoidVec (x #> th)) - y)
       in case maybeRegParam of
            Nothing -> newTh / fromIntegral m
            Just l ->
              let (t:ts) = toList theta
                  penalizedTh = fromList $ 0 : ((* (l / fromIntegral m)) <$> ts)
               in (newTh + penalizedTh) / fromIntegral m
