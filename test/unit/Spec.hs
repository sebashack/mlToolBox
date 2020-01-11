import Test.Tasty

import qualified Regression.CommonSpec as RegressionCommon
import qualified Regression.LinearSpec as RegressionLinear
import qualified Regression.LogisticSpec as RegressionLogistic

main :: IO ()
main = do
  linearRegressionTests <- RegressionLinear.tests
  commonRegressionTests <- RegressionCommon.tests
  logisticRegressionTests <- RegressionLogistic.tests
  defaultMain $
    testGroup
      "Main"
      [linearRegressionTests, commonRegressionTests, logisticRegressionTests]
