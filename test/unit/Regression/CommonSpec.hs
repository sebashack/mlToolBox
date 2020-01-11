module Regression.CommonSpec where

import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import Reexports (Matrix, R)
import Regression.Common (featureNormalize)
import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree, testGroup)
import Utils (matrixEq, readLinearRegressionSample)

tests :: IO TestTree
tests = do
  curDir <- getCurrentDirectory
  let dataDir = curDir </> "testData"
  (linearRegressionMatrix, normalizedLrMatrix, _) <-
    readLinearRegressionSample dataDir
  specs <-
    concat <$>
    mapM
      testSpecs
      [featureNormalizeSpec linearRegressionMatrix normalizedLrMatrix]
  return $
    testGroup
      "CommonRegression"
      [testGroup "Common Regression utilities specs" specs]

featureNormalizeSpec :: Matrix R -> Matrix R -> Spec
featureNormalizeSpec features expectedMatrix =
  describe "featureNormalize" $ do
    it "correctly normalizes the matrix values" $ do
      let normalizedMatrix = featureNormalize features
      normalizedMatrix `shouldSatisfy` (matrixEq expectedMatrix)
