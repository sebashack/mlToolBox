module Regression.LinearSpec where

import Hedgehog (Property, assert, forAll, property)
import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree)
import Test.Tasty (testGroup)
import Test.Tasty.Hedgehog (testProperty)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import ToolBox
  ( Matrix
  , R
  , Vector
  , computeCost
  , featureNormalize
  , getDimensions
  , gradientDescent
  , toVector
  )
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
      let r = computeCost features values (toVector [0, 0, 0])
          expectedValue = 65591548106.45744
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [25, 26, 27]" $ do
      let r = computeCost features values (toVector [25, 26, 27])
          expectedValue = 47251185844.64893
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [1500, 227, 230]" $ do
      let r = computeCost features values (toVector [1500, 227, 230])
          expectedValue = 11433546085.01064
      r `shouldSatisfy` doubleEq expectedValue
    it "should compute correctly for theta vector [-15.03, -27.123, -59.675]" $ do
      let r = computeCost features values (toVector [-15.03, -27.123, -59.675])
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
            (toVector [0, 0, 0])
            0.1
            50
            Nothing
        expectedTheta = toVector [338658.24925, 104127.51560, -172.20533]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.01 and 500 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (toVector [0, 0, 0])
            0.01
            500
            Nothing
        expectedTheta = toVector [338175.98397, 103831.11737, 103.03073]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it
    "should compute correctly for theta starting at [0, 0, 0], alpha = 0.001 and 1000 iterations" $ do
    let theta =
          gradientDescent
            (featureNormalize features)
            values
            (toVector [0, 0, 0])
            0.001
            1000
            Nothing
        expectedTheta = toVector [215244.48211, 61233.08697, 20186.40938]
    theta `shouldSatisfy` (vectorEq expectedTheta)
  it "decreases the value of the cost function the more iterations are computed" $ do
    let normalizedFeatures = featureNormalize features
        computeTheta numIters =
          gradientDescent
            normalizedFeatures
            values
            (toVector [0, 0, 0])
            0.01
            numIters
            Nothing
        costFValues =
          computeCost normalizedFeatures values . computeTheta <$>
          (take 100 $ iterate (+ 10) 1)
    costFValues `shouldSatisfy` isDescending

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
            (toVector $ replicate (snd $ getDimensions matrix) 0)
            alpha
            numIters
            Nothing
        costFValues =
          computeCost normalizedMatrix values . computeTheta <$>
          (take 20 $ iterate (+ 5) 1)
    assert $ isDescending costFValues
