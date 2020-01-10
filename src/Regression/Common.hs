module Regression.Common
  ( splitMatrixOfSamples
  , addOnesColumn
  , toMatrix
  , toVector
  , toListMatrix
  , toListVector
  , getDimensions
  , featureNormalize
  , sigmoid
  ) where

import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , cols
  , fromList
  , fromLists
  , rows
  , toList
  , toLists
  , tr'
  )
import Numeric.LinearAlgebra.Devel (foldVector)

type ListMatrix = [[R]]

type ListVector = [R]

splitMatrixOfSamples :: ListMatrix -> (ListMatrix, ListVector)
splitMatrixOfSamples mx = foldr accumFeaturesAndVals ([], []) mx
  where
    splitList ls =
      let (features, val) = splitAt (length ls - 1) ls
       in (features, head val)
    accumFeaturesAndVals ls (features, vals) =
      let (fs, vs) = splitList ls
       in (fs : features, vs : vals)

toMatrix :: ListMatrix -> Matrix R
toMatrix = fromLists

toListMatrix :: Matrix R -> ListMatrix
toListMatrix = toLists

toVector :: ListVector -> Vector R
toVector = fromList

toListVector :: Vector R -> ListVector
toListVector = toList

addOnesColumn :: ListMatrix -> ListMatrix
addOnesColumn = fmap (1.0 :)

getDimensions :: Matrix R -> (Int, Int)
getDimensions mx = (rows mx, cols mx)

sigmoid :: R -> R
sigmoid z = 1 / (1 + (e ** (-z)))
  where
    e = exp 1

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
      let (sum, n) = foldl (\(v', n) v -> (v + v', n + 1)) (0, 0) vals
       in (sum / n, n)
    computeStd n mn vals =
      let sumOfSquares = foldl (\v' v -> ((v - mn) ^ 2) + v') 0 vals
       in sqrt $ sumOfSquares / (n - 1)
    computeMeanAndStd vals =
      let (mean, n) = computeMean vals
          std = computeStd n mean vals
       in (mean, std)
    normalize mean std val =
      if std == 0
        then mean
        else (val - mean) / std
