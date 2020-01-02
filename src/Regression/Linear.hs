module Regression.Linear where

import Numeric.LinearAlgebra (( #> ), (<.>), (?), add)
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , atIndex
  , cols
  , flatten
  , fromList
  , matrix
  , scalar
  , size
  , tr'
  )

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
