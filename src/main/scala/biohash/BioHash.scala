package biohash

import scala.util.Random

/** Configuration for [[BioHash]] training and encoding.
  *
  * BioHash expands inputs to `m` hidden units, learns weight rows online with
  * Hebbian/anti-Hebbian updates, and encodes each item via k-WTA sparsification
  * (exactly `k` active bits). Activity fraction is `k / m`.
  *
  * @param inputDim expected length of input feature vectors (paper: `d`)
  * @param m hash layer width, i.e. number of hidden units (paper: `m`)
  * @param k hash length: number of active +1 bits per code (paper: `k`)
  * @param p exponent for weight normalization and scoring metric (paper: `p`; `p = 2` → dot product)
  * @param learningRate step size for online weight updates
  * @param epochs number of full passes over the training set
  * @param antiWinnerRank 1-indexed rank of the anti-winner unit (paper: `r`; receives gain `-delta`)
  * @param delta anti-Hebbian strength for the anti-winner (paper: `Δ`; `0` disables repulsive updates)
  * @param seed random seed for weight initialization and training shuffle order
  * @param normalizeInputs whether to L2-normalize inputs before scoring and learning
  * @param renormalizeWeights whether to project updated weight rows back to unit p-norm
  */
final case class BioHashConfig private (
    inputDim: Int,
    m: Int,
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

  require(m > 0 && k > 0 && k <= m, "BioHashConfig: require 0 < k <= m")
  require(inputDim > 0, "BioHashConfig: inputDim must be positive")
  require(antiWinnerRank >= 1 && antiWinnerRank <= m, "BioHashConfig: antiWinnerRank out of range")
  require(p >= 1.0, "BioHashConfig: p must be >= 1")

  def activityFraction: Double = k.toDouble / m

object BioHashConfig:

  /** Creates [[BioHashConfig]] using descriptive parameter names. */
  def apply(
      inputDimension: Int,
      hashLayerWidth: Int,
      hashLength: Int,
      normExponent: Double = 2.0,
      learningRate: Double = 0.01,
      trainingEpochs: Int = 5,
      antiWinnerRank: Int = 2,
      antiHebbianStrength: Double = 0.0,
      randomSeed: Long = 42L,
      normalizeInputs: Boolean = false,
      renormalizeWeights: Boolean = true
  ): BioHashConfig =
    new BioHashConfig(
      inputDim = inputDimension,
      m = hashLayerWidth,
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

  /** Paper notation: short parameter names (`m`, `k`, `p`, `delta`, …). */
  def paper(
      inputDim: Int,
      m: Int,
      k: Int,
      p: Double = 2.0,
      learningRate: Double = 0.01,
      epochs: Int = 5,
      antiWinnerRank: Int = 2,
      delta: Double = 0.0,
      seed: Long = 42L,
      normalizeInputs: Boolean = false,
      renormalizeWeights: Boolean = true
  ): BioHashConfig =
    new BioHashConfig(
      inputDim = inputDim,
      m = m,
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

  /** Paper-style config: m = k / activityFraction */
  def withActivity(inputDim: Int, k: Int, activity: Double): BioHashConfig =
    val m = math.ceil(k / activity).toInt
    apply(inputDimension = inputDim, hashLayerWidth = m, hashLength = k)

  def mnistDefault(k: Int = 2): BioHashConfig =
    apply(inputDimension = 784, hashLayerWidth = (k / 0.05).toInt, hashLength = k)

  def cifarDefault(k: Int = 2): BioHashConfig =
    apply(inputDimension = 3072, hashLayerWidth = (k / 0.005).toInt, hashLength = k)

/** Learned BioHash encoder/trainer with online Hebbian/anti-Hebbian updates. */
final class BioHash(val config: BioHashConfig) extends HashEncoder:

  private val rng = Random(config.seed)

  /** Weight matrix W[mu][i], m rows x d cols */
  val weights: Array[Array[Double]] = initializeWeights()

  private def initializeWeights(): Array[Array[Double]] =
    Array.tabulate(config.m) { _ =>
      val row = Array.fill(config.inputDim)(rng.nextGaussian())
      VectorOps.normalizeInPlace(row, config.p)
      row
    }

  def preprocess(x: Array[Double]): Array[Double] =
    if config.normalizeInputs then VectorOps.l2NormalizeInput(x) else x

  def scores(x: Array[Double]): Array[Double] =
    val input = preprocess(x)
    VectorOps.scoresMatrix(weights, input, config.p)

  def trainStep(x: Array[Double]): Unit =
    val input = preprocess(x)
    val s = VectorOps.scoresMatrix(weights, input, config.p)
    val ranks = TopK.rankIndices(s)

    val winner = ranks(0)
    updateRow(winner, input, s(winner), gain = 1.0)

    if config.delta != 0.0 then
      val antiWinner = ranks(config.antiWinnerRank - 1)
      if antiWinner != winner then
        updateRow(antiWinner, input, s(antiWinner), gain = -config.delta)

  private def updateRow(mu: Int, x: Array[Double], score: Double, gain: Double): Unit =
    val row = weights(mu)
    val lr = config.learningRate * gain
    var i = 0
    while i < row.length do
      row(i) += lr * (x(i) - score * row(i))
      i += 1
    if config.renormalizeWeights then VectorOps.normalizeInPlace(row, config.p)

  def train(data: IndexedSeq[Array[Double]]): Unit =
    var epoch = 0
    while epoch < config.epochs do
      val shuffled = rng.shuffle(data.toList)
      shuffled.foreach(trainStep)
      epoch += 1

  def encode(x: Array[Double]): SparseHash =
    val topK = TopK.topKIndices(scores(x), config.k)
    SparseHash.fromTopK(topK)

object BioHash:

  def apply(config: BioHashConfig): BioHash = new BioHash(config)

  /** Restore a trained encoder from persisted weights. */
  def fromWeights(config: BioHashConfig, weights: Array[Array[Double]]): BioHash =
    require(weights.length == config.m, "BioHash.fromWeights: row count must match m")
    require(weights.forall(_.length == config.inputDim), "BioHash.fromWeights: column count must match inputDim")
    val bh = new BioHash(config)
    var mu = 0
    while mu < weights.length do
      System.arraycopy(weights(mu), 0, bh.weights(mu), 0, config.inputDim)
      mu += 1
    bh
