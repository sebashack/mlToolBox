module NeuralNetwork.BackForwardPropagationSpec where

import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import NeuralNetwork.BackForwardPropagation (computeCost)
import Reexports (Matrix, R, Vector)
import Utils (doubleEq, readBackForwardSample)

tests :: IO TestTree
tests = do
  curDir <- getCurrentDirectory
  let dataDir = curDir </> "testData"
  (pixelsMatrix, numbersVector, weights0Matrix, weights1Matrix) <-
    readBackForwardSample dataDir
  specs <-
    concat <$>
    mapM
      testSpecs
      [ costFunctionSpec
          pixelsMatrix
          numbersVector
          [weights0Matrix, weights1Matrix]
      ]
  return $
    testGroup
      "BackForwardPropagation"
      [testGroup "BackForward Propagation specs" specs]

-- Specs
costFunctionSpec :: Matrix R -> Vector R -> [Matrix R] -> Spec
costFunctionSpec pixelsMatrix numbersVector weights =
  describe "computeCost" $ do
    it "foo" $ do
      let r = computeCost 10 pixelsMatrix numbersVector weights
          expectedValue = 0.287629
      r `shouldSatisfy` doubleEq expectedValue
