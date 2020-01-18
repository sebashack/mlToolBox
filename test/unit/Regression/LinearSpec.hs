module Regression.LinearSpec where

import Hedgehog (Property, assert, forAll, property)
import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree)
import Test.Tasty (testGroup)
import Test.Tasty.Hedgehog (testProperty)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import Reexports (Matrix, R, Vector, fromList)
import Regression.Linear (computeCost, gradientDescent, minimizeBFGS2)

import Regression.Common (MinimizationOpts(..), featureNormalize, getDimensions)

import Utils
  ( doubleEq
  , genAlpha
  , genMatrixAndVals
  , isDescending
  , readLinearRegressionSample
  , vectorEq
  )

tests :: IO TestTree
tests = do
  curDir <- getCurrentDirectory
  let dataDir = curDir </> "testData"
  (linearRegressionMatrix, _, linearRegressionValues) <-
    readLinearRegressionSample dataDir
  specs <-
    concat <$>
    mapM
      testSpecs
      [ costFunctionSpec linearRegressionMatrix linearRegressionValues
      , gradientDescentSpec linearRegressionMatrix linearRegressionValues
      , gradientBFGS2Spec linearRegressionMatrix linearRegressionValues
      ]
  let properties =
        uncurry testProperty <$>
        [ ( "The cost fuction must decrease the more iterations gradient descent takes"
          , costFunctionDecreasesTheMoreIterationsOfGradientDescent)
        ]
  return $
    testGroup
      "LinearRegression"
      [ testGroup "Linear Regression specs" specs
      , testGroup "Linear Regression properties" properties
      ]

-- Specs
costFunctionSpec :: Matrix R -> Vector R -> Spec
costFunctionSpec features values =
  describe "computeCost" $ do
    it "should compute correctly for theta vector [0, 0, 0]" $ do
      let r = computeCost features values (fromList [0, 0, 0])
          expectedValue = 65591548106.45744
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [25, 26, 27]" $ do
      let r = computeCost features values (fromList [25, 26, 27])
          expectedValue = 47251185844.64893
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [1500, 227, 230]" $ do
      let r = computeCost features values (fromList [1500, 227, 230])
          expectedValue = 11433546085.01064
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [-15.03, -27.123, -59.675]" $ do
      let r = computeCost features values (fromList [-15.03, -27.123, -59.675])
          expectedValue = 88102482793.02190
      r `shouldSatisfy` doubleEq expectedValue

gradientDescentSpec :: Matrix R -> Vector R -> Spec
gradientDescentSpec features values = do
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.1 and 50 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (fromList [0, 0, 0])
            0.1
            50
            0
        expectedTheta = fromList [338658.24925, 104127.51560, -172.20533]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.01 and 500 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (fromList [0, 0, 0])
            0.01
            500
            0
        expectedTheta = fromList [338175.98397, 103831.11737, 103.03073]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.001 and 1000 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (fromList [0, 0, 0])
            0.001
            1000
            0
        expectedTheta = fromList [215244.48211, 61233.08697, 20186.40938]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it "decreases the value of the cost function the more iterations are computed" $ do
    let normalizedFeatures = featureNormalize features
        computeTheta numIters =
          gradientDescent
            normalizedFeatures
            values
            (fromList [0, 0, 0])
            0.01
            numIters
            0
        costFValues =
          computeCost normalizedFeatures values . computeTheta <$>
          (take 100 $ iterate (+ 10) 1)
    costFValues `shouldSatisfy` isDescending

gradientBFGS2Spec :: Matrix R -> Vector R -> Spec
gradientBFGS2Spec features values = do
  it "should compute correctly for theta starting at [0, 0, 0], 500 iterations" $ do
    let theta =
          minimizeBFGS2
            (featureNormalize features)
            values
            (fromList [0, 0, 0])
            500
            (MinimizationOpts 0.01 0.1 0.01)
            0
        expectedTheta = fromList [340412.659, 110631.0502, -6649.474]
    theta `shouldSatisfy` (vectorEq expectedTheta)

-- Properties
costFunctionDecreasesTheMoreIterationsOfGradientDescent :: Property
costFunctionDecreasesTheMoreIterationsOfGradientDescent =
  property $ do
    (matrix, values) <- forAll genMatrixAndVals
    alpha <- forAll genAlpha
    let normalizedMatrix = featureNormalize matrix
        computeTheta numIters =
          gradientDescent
            normalizedMatrix
            values
            (fromList $ replicate (snd $ getDimensions matrix) 0)
            alpha
            numIters
            0
        costFValues =
          computeCost normalizedMatrix values . computeTheta <$>
          (take 20 $ iterate (+ 5) 1)
    assert $ isDescending costFValues
