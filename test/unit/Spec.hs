import Test.Tasty

import qualified Regression.CommonSpec as RegressionCommon
import qualified Regression.LinearSpec as RegressionLinear

main :: IO ()
main = do
  linearRegressionTests <- RegressionLinear.tests
  commonRegressionTests <- RegressionCommon.tests
  defaultMain $ testGroup "Main" [linearRegressionTests, commonRegressionTests]
