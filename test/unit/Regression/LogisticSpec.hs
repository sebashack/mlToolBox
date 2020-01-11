module Regression.LogisticSpec where

import System.Directory (getCurrentDirectory)
import System.FilePath.Posix ((</>))
import Test.Tasty (TestTree)
import Test.Tasty (testGroup)
import Test.Tasty.Hspec (Spec, describe, it, shouldSatisfy, testSpecs)

import Utils (readLogisticRegressionSample)

tests :: IO TestTree
tests = do
  curDir <- getCurrentDirectory
  let dataDir = curDir </> "testData"
  (logisticRegressionMatrix, linearRegressionValues) <-
    readLogisticRegressionSample dataDir
  specs <- concat <$> mapM testSpecs []
  return $
    testGroup "LogisticRegression" [testGroup "Logistic Regression specs" specs]

-- Specs
--costFunctionSpec :: Matrix R -> Vector R -> Spec
--costFunctionSpec features values =
--  describe "computeCost" $ do
--    it "should compute correctly for theta vector [0, 0, 0]" $ do
--      let r = computeCost features values (toVector [0, 0, 0])
--          expectedValue = 65591548106.45744
--      r `shouldSatisfy` doubleEq expectedValue
