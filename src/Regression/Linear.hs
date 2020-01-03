module Regression.Linear
  ( computeCostFunction
  , gradientDescent
  , addOnesColumn
  , splitMatrixOfSamples
  , toMatrix
  , toVector
  , featureNormalize
  , toListMatrix
  , toListVector
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
  , scalar
  , size
  , toList
  , toLists
  , tr'
  )

type ListMatrix = [[R]]

type ListVector = [R]

computeCostFunction :: Matrix R -> Vector R -> Vector R -> R
computeCostFunction x y theta =
  let m = fromIntegral $ size y
      a = add (x #> theta) (-1 * y)
   in (a <.> a) / (2 * m)

gradientDescent :: Matrix R -> Vector R -> Vector R -> R -> Int -> Vector R
gradientDescent x y theta alpha depth = go 0 theta
  where
    go k th
      | k >= depth = th
      | otherwise =
        let delta =
              computeDelta
                (fromIntegral $ size y)
                th
                0
                (fromList $ replicate (cols x) 0)
            th' = th - ((scalar alpha) * delta)
         in go (k + 1) th'
    --
    computeDelta m th i delta
      | i >= m = delta / (scalar $ fromIntegral m)
      | otherwise =
        let xVals = flatten $ x ? [i]
            delta' =
              add delta ((scalar $ (th <.> xVals) - (y `atIndex` i)) * xVals)
         in computeDelta m th (i + 1) delta'

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

splitMatrixOfSamples :: ListMatrix -> (ListMatrix, ListVector)
splitMatrixOfSamples mx = foldr accumFeaturesAndVals ([], []) mx
  where
    splitList ls =
      let (features, val) = splitAt (length ls - 1) ls
       in (features, head val)
    accumFeaturesAndVals ls (features, vals) =
      let (fs, vs) = splitList ls
       in (fs : features, vs : vals)
