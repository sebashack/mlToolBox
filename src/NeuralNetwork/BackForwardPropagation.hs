{-# LANGUAGE RecordWildCards #-}

module NeuralNetwork.BackForwardPropagation
  ( genRandomThetas
  , computeCost
  , minimizeBFGS2
  ) where

import Control.Monad (mapM)
import Numeric.GSL.Minimization (MinimizeMethodD(VectorBFGS2), minimizeVD)
import Numeric.LinearAlgebra (( #> ), (<.>), (<>), sumElements)
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , (===)
  , (?)
  , asColumn
  , atIndex
  , cmap
  , flatten
  , fromList
  , fromLists
  , matrix
  , reshape
  , size
  , takesV
  , tr'
  , vjoin
  )
import Numeric.LinearAlgebra.Devel (mapMatrixWithIndex)
import Numeric.Natural (Natural)
import Prelude hiding ((<>))
import System.Random (randomRIO)

import Common
  ( MinimizationOpts(..)
  , addOnesColumn
  , getDimensions
  , sigmoid
  , sigmoidMatrix
  )
import NeuralNetwork.Types (Layer(numUnits), Network(..))

genRandomThetas :: R -> Network -> IO [Matrix R]
genRandomThetas epsilon Network {..} =
  let allLayers = inputLayer : hiddenLayer ++ [outputLayer]
   in mapM mkTheta (zip allLayers (tail allLayers))
  where
    mkTheta :: (Layer, Layer) -> IO (Matrix R)
    mkTheta (l1, l2) = do
      let numRows = numUnits l2
          numCols = numUnits l1 + 1
          numElts = fromIntegral $ numRows * numCols
      elts <- sequence $ replicate numElts (randomRIO (0.0, 1.0))
      return $
        matrix
          (fromIntegral numCols)
          ((\r -> r * (2 * epsilon) - epsilon) <$> elts)

computeCost :: Int -> Matrix R -> Vector R -> [Matrix R] -> R
computeCost numLabels x y thetas = cost
  where
    m = size y
    --
    hypotheses =
      foldl (\an th -> sigmoidMatrix ((addOnesColumn an) <> (tr' th))) x thetas
    --
    cost =
      let rowElemsCost i k r =
            let yik =
                  if (round $ y `atIndex` i) == k
                    then 1.0
                    else 0.0
             in (-yik * (log (r))) - ((1 - yik) * log (1 - r))
          summation =
            sumElements $
            mapMatrixWithIndex (\(i, k) r -> rowElemsCost i k r) hypotheses
       in summation / fromIntegral m

minimizeBFGS2 ::
     Matrix R
  -> Vector R
  -> Int
  -> [Matrix R]
  -> Int
  -> MinimizationOpts
  -> Int
  -> [Matrix R]
minimizeBFGS2 x y numLabels thetas numIters MinimizationOpts {..} regFactor =
  let (flattenedParams, dimensions) = flattenParameters thetas
      (minParams, _) =
        minimizeVD
          VectorBFGS2
          precision
          numIters
          sizeOfFirstTrialStep
          tolerance
          (computeCost' dimensions)
          (computeGradients' dimensions)
          flattenedParams
   in unflattenParameters dimensions minParams
  where
    m = size y
    --
    flattenParameters :: [Matrix R] -> (Vector R, [(Int, Int)])
    flattenParameters mtxs =
      let (vecs, dims) =
            unzip $ (\mx -> (flatten mx, getDimensions mx)) <$> mtxs
       in (vjoin vecs, dims)
    --
    unflattenParameters :: [(Int, Int)] -> Vector R -> [Matrix R]
    unflattenParameters dims params =
      let sizes = uncurry (*) <$> dims
       in zipWith
            (\vec (ncols, _) -> reshape ncols vec)
            (takesV sizes params)
            dims
    --
    computeCost' :: [(Int, Int)] -> Vector R -> R
    computeCost' dims params =
      computeCost numLabels x y (unflattenParameters dims params)
    --
    computeGradients' :: [(Int, Int)] -> Vector R -> Vector R
    computeGradients' dims gradients =
      fst $
      flattenParameters $
      computeGradients 0 (unflattenParameters dims gradients)
    --
    computeActivationVals :: Matrix R -> [Matrix R]
    computeActivationVals a0 =
      let f vals@[] th = sigmoidMatrix (th * (addBiasUnit a0)) : vals
          f vals@(an:_) th = sigmoidMatrix (th * (addBiasUnit an)) : vals
       in foldl f [] thetas
    --
    computeErrors :: Matrix R -> [Matrix R] -> [Matrix R]
    computeErrors dL activationVals =
      let f vals@(dl:_) (an, th) = ((tr' th) * dl) .* an .* (1 - an) : vals
       in foldl f [dL] (zip activationVals (reverse thetas))
    --
    computeGradients :: Int -> [Matrix R] -> [Matrix R]
    computeGradients i gradients
      | i >= m = gradients
      | otherwise =
        let xi = (asColumn . flatten) $ x ? [i]
            a0 = xi
            activationVals@(aL:_) = computeActivationVals a0
            yi = round $ y `atIndex` i
            binEq :: Int -> R
            binEq n =
              if yi == n
                then 1.0
                else 0.0
            yik = asColumn $ fromList (binEq <$> [0 .. (numLabels - 1)])
            errorVals = computeErrors (aL - yik) (tail activationVals)
            accumGradient grad errVal actVal = grad + (errVal * (tr' actVal))
         in computeGradients (i + 1) $
            zipWith3
              accumGradient
              gradients
              errorVals
              (a0 : reverse activationVals)

elemWiseMult :: Matrix R -> Matrix R -> Matrix R
elemWiseMult mx1 mx2 =
  mapMatrixWithIndex (\idx r -> (mx1 `atIndex` idx) * r) mx2

(.*) :: Matrix R -> Matrix R -> Matrix R
(.*) = elemWiseMult

infixr 8 .*

-- Helpers
addBiasUnit :: Matrix R -> Matrix R
addBiasUnit mx = fromLists [[1]] === mx

addBiasUnit' :: Vector R -> Vector R
addBiasUnit' vec = vjoin [fromList [1], vec]
