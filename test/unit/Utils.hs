module Utils where

import qualified Data.ByteString.Lazy as LB (readFile)
import Data.Csv (HasHeader(NoHeader), decode)
import Data.Either (fromRight)
import qualified Data.Vector as V (empty, toList)
import Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import System.FilePath.Posix ((</>))

import Reexports (Matrix, R, Vector)

import Regression.Common
  ( addOnesColumn
  , splitMatrixOfSamples
  , toListMatrix
  , toListVector
  , toMatrix
  , toVector
  )

doubleEq :: Double -> Double -> Bool
doubleEq r1 r2 = abs (r1 - r2) <= 0.0001

matrixEq :: Matrix R -> Matrix R -> Bool
matrixEq mx1 mx2 =
  let valsMx1 = concat $ toListMatrix mx1
      valsMx2 = concat $ toListMatrix mx2
   in all (\(v1, v2) -> doubleEq v1 v2) $ zip valsMx1 valsMx2

vectorEq :: Vector R -> Vector R -> Bool
vectorEq vec1 vec2 =
  all (\(v1, v2) -> doubleEq v1 v2) $
  zip (toListVector vec1) (toListVector vec2)

isDescending :: [R] -> Bool
isDescending [] = True
isDescending [_] = True
isDescending (x1:x2:xs) = x1 >= x2 && isDescending (x2 : xs)

genAlpha :: Gen R
genAlpha = Gen.double (Range.constant 0.001 0.1)

genMatrixAndVals :: Gen (Matrix R, Vector R)
genMatrixAndVals = do
  numRows <- Gen.int (Range.linear 2 100)
  numColumns <- Gen.int (Range.linear 2 100)
  resultVectorVals <- genValues numRows 0 (pure [])
  matrixVals <- genMatrix numRows numColumns 0 (pure [])
  return $ (toMatrix . addOnesColumn $ matrixVals, toVector resultVectorVals)
  where
    genValues :: Int -> Int -> Gen [R] -> Gen [R]
    genValues n i accum =
      if i >= n
        then accum
        else do
          newVal <- Gen.double (Range.linearFrac 1.0 100000.0)
          accum' <- accum
          genValues n (i + 1) (pure $ newVal : accum')
    --
    genMatrix :: Int -> Int -> Int -> Gen [[R]] -> Gen [[R]]
    genMatrix rows cols i accum =
      if i >= rows
        then accum
        else do
          newRow <- genValues cols 0 (pure [])
          accum' <- accum
          genMatrix rows cols (i + 1) (pure $ newRow : accum')

readLinearRegressionSample :: FilePath -> IO (Matrix R, Matrix R, Vector R)
readLinearRegressionSample dataDir = do
  let linearRegressionFile = dataDir </> "linearRegression.csv"
  linearRegressionData <- decode NoHeader <$> LB.readFile linearRegressionFile
  let (features, vals) =
        splitMatrixOfSamples . V.toList . fromRight V.empty $
        linearRegressionData
  let normalizedDataFile = dataDir </> "lrFeatureNormalize.csv"
  normalizedLrData <- decode NoHeader <$> LB.readFile normalizedDataFile
  let normalizedFeatures =
        toMatrix . addOnesColumn . V.toList . fromRight V.empty $
        normalizedLrData
  return
    (toMatrix . addOnesColumn $ features, normalizedFeatures, toVector vals)

readLogisticRegressionSample :: FilePath -> IO (Matrix R, Vector R)
readLogisticRegressionSample dataDir = do
  let logisticRegressionFile = dataDir </> "logisticRegression.csv"
  logisticRegressionData <-
    decode NoHeader <$> LB.readFile logisticRegressionFile
  let (features, vals) =
        splitMatrixOfSamples . V.toList . fromRight V.empty $
        logisticRegressionData
  return (toMatrix . addOnesColumn $ features, toVector vals)
