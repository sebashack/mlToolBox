module Regression.Linear
  ( computeCost
  , gradientDescent
  , addOnesColumn
  , splitMatrixOfSamples
  , toMatrix
  , toVector
  , featureNormalize
  , toListMatrix
  , toListVector
  , getDimensions
  ) where

import Numeric.LinearAlgebra (( #> ), (<.>), (?), add)
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , atIndex
  , cols
  , flatten
  , fromList
  , fromLists
  , matrix
  , rows
  , scalar
  , size
  , toList
  , toLists
  , tr'
  )

type ListMatrix = [[R]]

type ListVector = [R]

computeCost :: Matrix R -> Vector R -> Vector R -> R
computeCost x y theta =
  let m = fromIntegral $ size y
      a = add (x #> theta) (-1 * y)
   in (a <.> a) / (2 * m)

gradientDescent ::
     Matrix R -> Vector R -> Vector R -> R -> Int -> Maybe R -> Vector R
gradientDescent x y theta alpha depth maybeRegParam = go 0 theta
  where
    m :: Int
    m = size y
    --
    go k accum
      | k >= depth = accum
      | otherwise =
        let delta = computeDelta accum 0 (fromList $ replicate (cols x) 0)
            accum' =
              if k == 0
                then accum
                else accum * regFactor
         in go (k + 1) (accum' - ((scalar alpha) * delta))
    --
    computeDelta :: Vector R -> Int -> Vector R -> Vector R
    computeDelta th i delta
      | i >= m = delta / (scalar $ fromIntegral m)
      | otherwise =
        let xVals = flatten $ x ? [i]
            delta' =
              add delta ((scalar $ (th <.> xVals) - (y `atIndex` i)) * xVals)
         in computeDelta th (i + 1) delta'
    --
    regFactor =
      maybe
        (scalar (1 :: R))
        (\lambda -> scalar $ 1 - ((alpha * lambda) / fromIntegral m))
        maybeRegParam

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

splitMatrixOfSamples :: ListMatrix -> (ListMatrix, ListVector)
splitMatrixOfSamples mx = foldr accumFeaturesAndVals ([], []) mx
  where
    splitList ls =
      let (features, val) = splitAt (length ls - 1) ls
       in (features, head val)
    accumFeaturesAndVals ls (features, vals) =
      let (fs, vs) = splitList ls
       in (fs : features, vs : vals)
