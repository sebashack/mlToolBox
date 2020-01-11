module Regression.LogisticSpec where

import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree)
import Test.Tasty (testGroup)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import Reexports (Matrix, R, Vector)
import Regression.Common (featureNormalize, toVector)
import Regression.Logistic (computeCost, gradientDescent)
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
      ]
  return $
    testGroup "LogisticRegression" [testGroup "Logistic Regression specs" specs]

-- Specs
costFunctionSpec :: Matrix R -> Vector R -> Spec
costFunctionSpec features values =
  describe "computeCost" $ do
    it "should compute correctly for theta vector [0, 0, 0]" $ do
      let r = computeCost features values (toVector [0, 0, 0])
          expectedValue = 0.693147
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [-24, 0.2, 0.2]" $ do
      let r = computeCost features values (toVector [-24, 0.2, 0.2])
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
            (toVector [0, 0, 0])
            0.01
            165000
            Nothing
        expectedTheta = toVector [1.718447, 4.012899, 3.743847]
        r = computeCost (featureNormalize features) values theta
        expectedValue = 0
    theta `shouldSatisfy` (vectorEq expectedTheta)
