package biohash

import scala.util.Random

/** Configuration for [[NaiveBioHash]] training and encoding.
  *
  * NaiveBioHash is an ablation baseline: it uses the same Hebbian/anti-Hebbian
  * learning rule as [[BioHash]] but projects into `k` dense hidden units (no
  * expansive k-WTA over `m > k` units) and binarizes by activation sign.
  *
  * @param inputDim expected length of input feature vectors (paper: `d`)
  * @param k hash length: number of dense hidden units and sign bits (paper: `k`)
  * @param p exponent for weight normalization and scoring metric (paper: `p`; `p = 2` → dot product)
  * @param learningRate step size for online weight updates
  * @param epochs number of full passes over the training set
  * @param antiWinnerRank 1-indexed rank of the anti-winner unit (paper: `r`; receives gain `-delta`)
  * @param delta anti-Hebbian strength for the anti-winner (paper: `Δ`; `0` disables repulsive updates)
  * @param seed random seed for weight initialization and training shuffle order
  * @param normalizeInputs whether to L2-normalize inputs before scoring and learning
  * @param renormalizeWeights whether to project updated weight rows back to unit p-norm
  */
final case class NaiveBioHashConfig private (
    inputDim: Int,
    k: Int,
    p: Double = 2.0,
    learningRate: Double = 0.01,
    epochs: Int = 5,
    antiWinnerRank: Int = 2,
    delta: Double = 0.0,
    seed: Long = 42L,
    normalizeInputs: Boolean = false,
    renormalizeWeights: Boolean = true
):

  require(k > 0 && inputDim > 0)

object NaiveBioHashConfig:

  /** Creates [[NaiveBioHashConfig]] using descriptive parameter names. */
  def apply(
      inputDimension: Int,
      hashLength: Int,
      normExponent: Double = 2.0,
      learningRate: Double = 0.01,
      trainingEpochs: Int = 5,
      antiWinnerRank: Int = 2,
      antiHebbianStrength: Double = 0.0,
      randomSeed: Long = 42L,
      normalizeInputs: Boolean = false,
      renormalizeWeights: Boolean = true
  ): NaiveBioHashConfig =
    new NaiveBioHashConfig(
      inputDim = inputDimension,
      k = hashLength,
      p = normExponent,
      learningRate = learningRate,
      epochs = trainingEpochs,
      antiWinnerRank = antiWinnerRank,
      delta = antiHebbianStrength,
      seed = randomSeed,
      normalizeInputs = normalizeInputs,
      renormalizeWeights = renormalizeWeights
    )

  /** Paper notation: short parameter names (`k`, `p`, `delta`, …). */
  def paper(
      inputDim: Int,
      k: Int,
      p: Double = 2.0,
      learningRate: Double = 0.01,
      epochs: Int = 5,
      antiWinnerRank: Int = 2,
      delta: Double = 0.0,
      seed: Long = 42L,
      normalizeInputs: Boolean = false,
      renormalizeWeights: Boolean = true
  ): NaiveBioHashConfig =
    new NaiveBioHashConfig(
      inputDim = inputDim,
      k = k,
      p = p,
      learningRate = learningRate,
      epochs = epochs,
      antiWinnerRank = antiWinnerRank,
      delta = delta,
      seed = seed,
      normalizeInputs = normalizeInputs,
      renormalizeWeights = renormalizeWeights
    )

/** NaiveBioHash ablation: k dense hidden units, same learning rule, sign binarization (no expansive k-WTA). */
final class NaiveBioHash(val config: NaiveBioHashConfig) extends HashEncoder:

  private val bioConfig = BioHashConfig.paper(
    inputDim = config.inputDim,
    m = config.k,
    k = config.k,
    p = config.p,
    learningRate = config.learningRate,
    epochs = config.epochs,
    antiWinnerRank = math.min(config.antiWinnerRank, config.k),
    delta = config.delta,
    seed = config.seed,
    normalizeInputs = config.normalizeInputs,
    renormalizeWeights = config.renormalizeWeights
  )

  private val inner = new BioHash(bioConfig)

  def train(data: IndexedSeq[Array[Double]]): Unit = inner.train(data)

  /** Weight matrix used for artifact persistence. */
  def savedWeights: Array[Array[Double]] = inner.weights

  /** Encode by sign: active where score > 0 (up to k bits; may be fewer if some scores <= 0). */
  def encode(x: Array[Double]): SparseHash =
    val s = inner.scores(x)
    val active = s.zipWithIndex.collect { case (score, idx) if score > 0 => idx }.sorted
    SparseHash(active, active.length)

object NaiveBioHash:

  def apply(config: NaiveBioHashConfig): NaiveBioHash = new NaiveBioHash(config)

  /** Restore a trained NaiveBioHash encoder from persisted weights. */
  def fromWeights(config: NaiveBioHashConfig, weights: Array[Array[Double]]): NaiveBioHash =
    val innerConfig = BioHashConfig.paper(
      inputDim = config.inputDim,
      m = config.k,
      k = config.k,
      p = config.p,
      learningRate = config.learningRate,
      epochs = config.epochs,
      antiWinnerRank = math.min(config.antiWinnerRank, config.k),
      delta = config.delta,
      seed = config.seed,
      normalizeInputs = config.normalizeInputs,
      renormalizeWeights = config.renormalizeWeights
    )
    val nb = new NaiveBioHash(config)
    val restored = BioHash.fromWeights(innerConfig, weights)
    var mu = 0
    while mu < weights.length do
      System.arraycopy(restored.weights(mu), 0, nb.savedWeights(mu), 0, config.inputDim)
      mu += 1
    nb
