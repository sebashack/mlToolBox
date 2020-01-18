{-# LANGUAGE RecordWildCards #-}

module NeuralNetwork.BackForwardPropagation where

import Control.Monad (mapM)
import Numeric.LinearAlgebra ((<.>))
import Numeric.LinearAlgebra.Data
  ( Matrix
  , R
  , Vector
  , (===)
  , (?)
  , asColumn
  , atIndex
  , flatten
  , fromList
  , fromLists
  , matrix
  , size
  )
import Numeric.LinearAlgebra.Data (cmap)
import Numeric.Natural (Natural)
import System.Random (randomRIO)

import Common (MinimizationOpts(..), sigmoid, sigmoidMatrix)
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
computeCost numLabels x y thetas = go 0 0
  where
    m = size y
    --
    computeHypothesis :: Matrix R -> Matrix R
    computeHypothesis a0 =
      foldl (\an th -> sigmoidMatrix (th * (addBiasUnit an))) a0 thetas
    --
    addBiasUnit :: Matrix R -> Matrix R
    addBiasUnit m = fromLists [[1]] === m
    --
    go :: Int -> R -> R
    go i cost
      | i >= m = cost
      | otherwise =
        let xi = (asColumn . flatten) $ x ? [i]
            hypothesisVec = flatten $ computeHypothesis xi
            yi = round $ y `atIndex` i
            binEq :: Int -> R
            binEq n =
              if yi == n
                then 1.0
                else 0.0
            yik = fromList (binEq <$> [0 .. (numLabels - 1)])
            a = cmap log hypothesisVec
            b = cmap log (1 - hypothesisVec)
            v = (((-1 * yik) <.> a) - ((1 - yik) <.> b))
         in go (i + 1) (cost + v)

minimizeBFGS2 ::
     Matrix R
  -> Vector R
  -> [Matrix R]
  -> Int
  -> MinimizationOpts
  -> Int
  -> Matrix R
minimizeBFGS2 x y thetas numIters MinimizationOpts {..} regFactor = undefined
