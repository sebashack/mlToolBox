module Regression.Linear
  ( computeCost
  , gradientDescent
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
