
{-# LANGUAGE FlexibleContexts #-}

module Common
  ( splitMatrixOfSamples
  , addOnesColumn
  , getDimensions
  , featureNormalize
  , sigmoid
  , sigmoidMatrix
  , regularizeCost
  , MinimizationOpts(..)
  ) where

import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , (|||)
  , (¿)
  , cmap
  , cols
  , flatten
  , fromList
  , fromLists
  , matrix
  , rows
  , toList
  , toLists
  , tr'
  )
import Numeric.LinearAlgebra (Container)
import Numeric.LinearAlgebra.Devel (foldVector)

data MinimizationOpts = MinimizationOpts
  { precision :: R
  , tolerance :: R
  , sizeOfFirstTrialStep :: R
  } deriving (Eq)

splitMatrixOfSamples :: Matrix R -> (Matrix R, Vector R)
splitMatrixOfSamples mx =
  let c = cols mx
   in (mx ¿ [0 .. (c - 2)], flatten $ mx ¿ [(c - 1)])

addOnesColumn :: Matrix R -> Matrix R
addOnesColumn mx = matrix 1 (replicate (rows mx) 1) ||| mx

getDimensions :: Matrix R -> (Int, Int)
getDimensions mx = (rows mx, cols mx)

sigmoid :: R -> R
sigmoid z = 1 / (1 + exp (-1 * z))

sigmoidMatrix :: Container c R => c R -> c R
sigmoidMatrix = cmap sigmoid

regularizeCost :: Int -> Vector R -> R -> R -> R
regularizeCost m theta lambda cost =
  let m' = fromIntegral m
      regVal = (foldVector (\v accum -> (v ** 2) + accum) 0 theta)
   in cost + ((regVal * lambda) / (3 * m'))

featureNormalize :: Matrix R -> Matrix R
featureNormalize mx =
  let trMx = toLists $ tr' mx
      normalizedTrMx =
        zipWith
          (\vals (mean, std) -> normalize mean std <$> vals)
          trMx
          (computeMeanAndStd <$> trMx)
   in tr' $ fromLists normalizedTrMx
  where
    computeMean vals =
      let (accum, n) = foldl (\(v', n') v -> (v + v', n' + 1)) (0, 0) vals
       in (accum / n, n)
    computeStd n mn vals =
      let sumOfSquares =
            foldl (\v' v -> ((v - mn) ** (2 :: Double)) + v') (0 :: Double) vals
       in sqrt $ sumOfSquares / (n - 1)
    computeMeanAndStd vals =
      let (mean, n) = computeMean vals
          std = computeStd n mean vals
       in (mean, std)
    normalize mean std val =
      if std == 0
        then mean
        else (val - mean) / std
