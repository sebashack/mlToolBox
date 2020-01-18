module Regression.LogisticSpec where

import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree)
import Test.Tasty (testGroup)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import Common (MinimizationOpts(..), featureNormalize)
import Reexports (Matrix, R, Vector, fromList)
import Regression.Logistic (computeCost, gradientDescent, minimizeBFGS2)
import Utils (doubleEq, readLogisticRegressionSample, vectorEq)

tests :: IO TestTree
tests = do
  curDir <- getCurrentDirectory
  let dataDir = curDir </> "testData"
  (logisticRegressionMatrix, logisticRegressionValues) <-
    readLogisticRegressionSample dataDir
  specs <-
    concat <$>
    mapM
      testSpecs
      [ costFunctionSpec logisticRegressionMatrix logisticRegressionValues
      , gradientDescentSpec logisticRegressionMatrix logisticRegressionValues
      , gradientBFGS2Spec logisticRegressionMatrix logisticRegressionValues
      ]
  return $
    testGroup "LogisticRegression" [testGroup "Logistic Regression specs" specs]

-- Specs
costFunctionSpec :: Matrix R -> Vector R -> Spec
costFunctionSpec features values =
  describe "computeCost" $ do
    it "should compute correctly for theta vector [0, 0, 0]" $ do
      let r = computeCost features values (fromList [0, 0, 0])
          expectedValue = 0.693147
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [-24, 0.2, 0.2]" $ do
      let r = computeCost features values (fromList [-24, 0.2, 0.2])
          expectedValue = 0.21833
      r `shouldSatisfy` doubleEq expectedValue

gradientDescentSpec :: Matrix R -> Vector R -> Spec
gradientDescentSpec features values = do
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.01 and 165000 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (fromList [0, 0, 0])
            0.01
            165000
            0
        expectedTheta = fromList [1.718447, 4.012899, 3.743847]
    theta `shouldSatisfy` (vectorEq expectedTheta)

gradientBFGS2Spec :: Matrix R -> Vector R -> Spec
gradientBFGS2Spec features values = do
  it
    "should compute correctly for theta starting at [0, 0, 0], and 400 iterations" $ do
    let theta =
          minimizeBFGS2
            features
            values
            (fromList [0, 0, 0])
            400
            (MinimizationOpts 0.01 0.1 0.01)
            0
        expectedTheta = fromList [-24.439, 0.2004, 0.1956]
    theta `shouldSatisfy` (vectorEq expectedTheta)
