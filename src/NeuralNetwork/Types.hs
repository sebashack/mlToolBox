module NeuralNetwork.Types where

import Numeric.LinearAlgebra.Data (R, Vector)
import Numeric.Natural (Natural)

newtype Layer = Layer
  { numUnits :: Natural
  } deriving (Eq, Show)

data Network = Network
  { inputLayer :: Layer
  , hiddenLayer :: [Layer]
  , outputLayer :: Layer
  } deriving (Eq, Show)
