import Test.Tasty
import Test.Tasty.Hspec

import qualified Regression.LinearSpec as RegressionLinear

main :: IO ()
main = do
  linearRegressionTests <- RegressionLinear.tests
  defaultMain $ testGroup "Main" [linearRegressionTests]
