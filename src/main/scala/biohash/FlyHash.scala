// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash

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

/** Random-projection FlyHash baseline: no learning, k-WTA encoding. */
final class FlyHash(val config: FlyHashConfig) extends HashEncoder:

  private val rng = Random(config.seed)

  /** Sparse random projection: each hidden unit connects to a random subset of inputs. */
  val weights: Array[Array[Double]] = initializeWeights()

  private def initializeWeights(): Array[Array[Double]] =
    val numConnections = math.max(1, (config.inputDim * config.samplingRate).toInt)
    Array.tabulate(config.m) { _ =>
      val row = Array.fill(config.inputDim)(0.0)
      val indices = rng.shuffle((0 until config.inputDim).toList).take(numConnections)
      indices.foreach { i =>
        row(i) = rng.nextGaussian()
      }
      VectorOps.normalizeInPlace(row, 2.0)
      row
    }

  def preprocess(x: Array[Double]): Array[Double] =
    if config.normalizeInputs then VectorOps.l2NormalizeInput(x) else x

  def scores(x: Array[Double]): Array[Double] =
    val input = preprocess(x)
    VectorOps.scoresMatrix(weights, input, p = 2.0)

  def encode(x: Array[Double]): SparseHash =
    val topK = TopK.topKIndices(scores(x), config.k)
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
      System.arraycopy(weights(mu), 0, fh.weights(mu), 0, config.inputDim)
      mu += 1
    fh
