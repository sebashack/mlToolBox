module Utils where

import qualified Data.ByteString.Lazy as LB (readFile)
import Data.Csv (HasHeader(NoHeader), decode)
import Data.Either (fromRight)
import qualified Data.Vector as V (empty, toList)
import Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import System.FilePath.Posix ((</>))

import Reexports (Matrix, R, Vector, fromLists, matrix, toList, toLists)

import Common (addOnesColumn, splitMatrixOfSamples)

doubleEq :: Double -> Double -> Bool
doubleEq r1 r2 = abs (r1 - r2) <= 0.001

matrixEq :: Matrix R -> Matrix R -> Bool
matrixEq mx1 mx2 =
  let valsMx1 = concat $ toLists mx1
      valsMx2 = concat $ toLists mx2
   in all (\(v1, v2) -> doubleEq v1 v2) $ zip valsMx1 valsMx2

vectorEq :: Vector R -> Vector R -> Bool
vectorEq vec1 vec2 =
  all (\(v1, v2) -> doubleEq v1 v2) $ zip (toList vec1) (toList vec2)

isDescending :: [R] -> Bool
isDescending [] = True
isDescending [_] = True
isDescending (x1:x2:xs) = x1 >= x2 && isDescending (x2 : xs)

genAlpha :: Gen R
genAlpha = Gen.double (Range.constant 0.001 0.1)

genMatrixAndVals :: Gen (Matrix R, Vector R)
genMatrixAndVals = do
  numRows <- Gen.int (Range.linear 2 100)
  numCols <- Gen.int (Range.linear 2 100)
  mxVals <- genValues (numRows * numCols) 0 (pure [])
  let mx = addOnesColumn $ matrix numCols mxVals
  return $ splitMatrixOfSamples mx
  where
    genValues :: Int -> Int -> Gen [R] -> Gen [R]
    genValues n i accum =
      if i >= n
        then accum
        else do
          newVal <- Gen.double (Range.linearFrac 1.0 100000.0)
          accum' <- accum
          genValues n (i + 1) (pure $ newVal : accum')

readLinearRegressionSample :: FilePath -> IO (Matrix R, Matrix R, Vector R)
readLinearRegressionSample dataDir = do
  let linearRegressionFile = dataDir </> "linearRegression.csv"
  linearRegressionData <- decode NoHeader <$> LB.readFile linearRegressionFile
  let (features, vals) =
        splitMatrixOfSamples .
        addOnesColumn . fromLists . V.toList . fromRight V.empty $
        linearRegressionData
  let normalizedDataFile = dataDir </> "lrFeatureNormalize.csv"
  normalizedLrData <- decode NoHeader <$> LB.readFile normalizedDataFile
  let normalizedFeatures =
        addOnesColumn . fromLists . V.toList . fromRight V.empty $
        normalizedLrData
  return (features, normalizedFeatures, vals)

readLogisticRegressionSample :: FilePath -> IO (Matrix R, Vector R)
readLogisticRegressionSample dataDir = do
  let logisticRegressionFile = dataDir </> "logisticRegression.csv"
  logisticRegressionData <-
    decode NoHeader <$> LB.readFile logisticRegressionFile
  let (features, vals) =
        splitMatrixOfSamples .
        addOnesColumn . fromLists . V.toList . fromRight V.empty $
        logisticRegressionData
  return (features, vals)
