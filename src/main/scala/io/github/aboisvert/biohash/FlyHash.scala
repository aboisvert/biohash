// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import scala.util.Random

final case class FlyHashConfig(
    inputDim: Int,
    m: Int,
    k: Int,
    samplingRate: Double = 0.1,
    seed: Long = 42L,
    normalizeInputs: Boolean = false
):

  require(m > 0 && k > 0 && k <= m)
  require(inputDim > 0)
  require(samplingRate > 0.0 && samplingRate <= 1.0)

object FlyHashConfig:

  /** Paper baseline: m = 10 * d with ~10% sampling (PN→KC). */
  def paperBaseline(inputDim: Int, k: Int, seed: Long = 42L): FlyHashConfig =
    FlyHashConfig(
      inputDim = inputDim,
      m = 10 * inputDim,
      k = k,
      samplingRate = 0.1,
      seed = seed
    )

/** Sparse random projection row: sorted input indices and corresponding weights. */
final case class SparseProjectionRow(indices: Array[Int], weights: Array[Double]):

  def score(x: Array[Double]): Double =
    var sum = 0.0
    var i = 0
    while i < indices.length do
      sum += weights(i) * x(indices(i))
      i += 1
    sum

/** Random-projection FlyHash baseline: no learning, k-WTA encoding. */
final class FlyHash(val config: FlyHashConfig) extends HashEncoder:

  private val rng = Random(config.seed)

  /** Sparse random projection rows; each hidden unit connects to a random input subset. */
  private var sparseRows: Array[SparseProjectionRow] = initializeSparseRows()

  /** Dense nested weights for artifact persistence and compatibility. */
  def weights: Array[Array[Double]] =
    sparseRows.map { row =>
      val dense = new Array[Double](config.inputDim)
      var i = 0
      while i < row.indices.length do
        dense(row.indices(i)) = row.weights(i)
        i += 1
      dense
    }

  private val scoreBuffer = new Array[Double](config.m)
  private val inputScratch = new Array[Double](config.inputDim)

  private def initializeSparseRows(): Array[SparseProjectionRow] =
    val numConnections = math.max(1, (config.inputDim * config.samplingRate).toInt)
    Array.tabulate(config.m) { _ =>
      val indices = rng.shuffle((0 until config.inputDim).toList).take(numConnections).sorted.toArray
      val weights = Array.tabulate(indices.length) { _ =>
        rng.nextGaussian()
      }
      val row = new Array[Double](weights.length)
      System.arraycopy(weights, 0, row, 0, weights.length)
      VectorOps.normalizeInPlace(row, 2.0)
      SparseProjectionRow(indices, row)
    }

  private def preprocess(x: Array[Double], dest: Array[Double]): Array[Double] =
    if config.normalizeInputs then
      System.arraycopy(x, 0, dest, 0, x.length)
      VectorOps.normalizeInPlace(dest, 2.0)
      dest
    else x

  def scores(x: Array[Double]): Array[Double] =
    val input = preprocess(x, inputScratch)
    val out = new Array[Double](config.m)
    scoresInto(input, out)
    out

  def scoresInto(input: Array[Double], out: Array[Double]): Unit =
    var row = 0
    while row < config.m do
      out(row) = sparseRows(row).score(input)
      row += 1

  def encode(x: Array[Double]): SparseHash =
    val input = preprocess(x, inputScratch)
    scoresInto(input, scoreBuffer)
    val topK = TopK.topKIndices(scoreBuffer, config.k)
    SparseHash.fromTopK(topK)

object FlyHash:

  def apply(config: FlyHashConfig): FlyHash = new FlyHash(config)

  /** Restore a FlyHash encoder from persisted weights. */
  def fromWeights(config: FlyHashConfig, weights: Array[Array[Double]]): FlyHash =
    require(weights.length == config.m, "FlyHash.fromWeights: row count must match m")
    require(weights.forall(_.length == config.inputDim), "FlyHash.fromWeights: column count must match inputDim")
    val fh = new FlyHash(config)
    var mu = 0
    while mu < weights.length do
      val dense = weights(mu)
      val indices = dense.indices.filter(dense(_) != 0.0).toArray
      val rowWeights = indices.map(dense(_))
      fh.sparseRows(mu) = SparseProjectionRow(indices, rowWeights)
      mu += 1
    fh
