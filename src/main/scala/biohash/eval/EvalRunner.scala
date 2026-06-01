// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash.eval

import biohash.*
import biohash.data.*

enum HashMethod:
  case BioHash, FlyHash, NaiveBioHash

/** Hyperparameters and method selection for an evaluation run via [[EvalRunner]]. */
final case class EvalConfig(
    /** Hash length: number of active bits per code (paper: `k`). */
    k: Int = 2,
    /** Activity fraction `a = k / m`; used to set hash layer width as `m = ceil(k / activity)`. */
    activity: Double = 0.05,
    /** Number of full training passes over the training split. */
    epochs: Int = 5,
    /** Step size for online weight updates (BioHash and NaiveBioHash). */
    learningRate: Double = 0.01,
    /** Anti-Hebbian strength for the anti-winner unit (paper: `Δ`; `0` disables repulsive updates). */
    delta: Double = 0.0,
    /** 1-indexed rank of the anti-winner unit (paper: `r`). */
    antiWinnerRank: Int = 2,
    /** Random seed for weight initialization and training shuffle order. */
    seed: Long = 42L,
    /** Whether to L2-normalize inputs before scoring and learning. */
    normalizeInputs: Boolean = false,
    /** Hashing method to train and evaluate. */
    method: HashMethod = HashMethod.BioHash,
    /** Override default retrieval cutoff; when unset, dataset-specific defaults apply. */
    retrievalLimit: Option[Int] = None
)

/** Metrics and timing from a single evaluation run. */
final case class EvalResult(
    /** Display name of the hashing method (e.g. `"BioHash"`, `"FlyHash"`). */
    method: String,
    /** Dataset identifier (e.g. `"mnist"`, `"cifar-10"`, `"sift10k"`). */
    dataset: String,
    /** Hash length used in the run (paper: `k`). */
    k: Int,
    /** Hash layer width derived from `k` and activity fraction (paper: `m`). */
    m: Int,
    /** Mean average precision for classification splits, or recall@R for ANN benchmarks. */
    mAP: Double,
    /** Wall-clock seconds spent training the encoder. */
    trainSeconds: Double,
    /** Wall-clock seconds spent encoding database and query vectors. */
    encodeSeconds: Double,
    /** Wall-clock seconds spent running retrieval and computing the metric. */
    querySeconds: Double,
    /** Maximum number of database items ranked per query when computing the metric. */
    retrievalLimit: Int
)

object EvalRunner:

  def runSplit(split: RetrievalSplit, config: EvalConfig, datasetName: String): EvalResult =
    val inputDim = split.trainVectors.head.length
    val m = math.ceil(config.k / config.activity).toInt
    val retrievalLimit = config.retrievalLimit.getOrElse:
      if datasetName.contains("cifar") then 1000
      else if datasetName.contains("synthetic-text") then math.min(100, split.databaseVectors.length)
      else split.databaseVectors.length

    val trainStart = System.nanoTime()
    val encoder: HashEncoder = config.method match
      case HashMethod.BioHash =>
        val bh = new BioHash(
          BioHashConfig.paper(
            inputDim = inputDim,
            m = m,
            k = config.k,
            learningRate = config.learningRate,
            epochs = config.epochs,
            antiWinnerRank = config.antiWinnerRank,
            delta = config.delta,
            seed = config.seed,
            normalizeInputs = config.normalizeInputs
          )
        )
        bh.train(split.trainVectors)
        bh
      case HashMethod.FlyHash =>
        new FlyHash(FlyHashConfig.paperBaseline(inputDim, config.k, config.seed))
      case HashMethod.NaiveBioHash =>
        val nb = new NaiveBioHash(
          NaiveBioHashConfig.paper(
            inputDim = inputDim,
            k = config.k,
            learningRate = config.learningRate,
            epochs = config.epochs,
            antiWinnerRank = config.antiWinnerRank,
            delta = config.delta,
            seed = config.seed,
            normalizeInputs = config.normalizeInputs
          )
        )
        nb.train(split.trainVectors)
        nb
    val methodName = config.method match
      case HashMethod.BioHash      => "BioHash"
      case HashMethod.FlyHash      => "FlyHash"
      case HashMethod.NaiveBioHash => "NaiveBioHash"
    val trainSeconds = (System.nanoTime() - trainStart) / 1e9

    val encodeStart = System.nanoTime()
    val dbHashes = encoder.encodeAll(split.databaseVectors)
    val queryHashes = encoder.encodeAll(split.queryVectors)
    val encodeSeconds = (System.nanoTime() - encodeStart) / 1e9

    val queryStart = System.nanoTime()
    val mAP = Metrics.evaluateRetrieval(
      queryHashes,
      dbHashes,
      split.queryLabels,
      split.databaseLabels,
      retrievalLimit
    )
    val querySeconds = (System.nanoTime() - queryStart) / 1e9

    EvalResult(
      method = methodName,
      dataset = datasetName,
      k = config.k,
      m = m,
      mAP = mAP,
      trainSeconds = trainSeconds,
      encodeSeconds = encodeSeconds,
      querySeconds = querySeconds,
      retrievalLimit = retrievalLimit
    )

  def sweepK(
      split: RetrievalSplit,
      ks: Seq[Int],
      baseConfig: EvalConfig,
      datasetName: String
  ): IndexedSeq[EvalResult] =
    ks.map(k => runSplit(split, baseConfig.copy(k = k), datasetName)).toIndexedSeq

  def runAnnDataset(
      ann: AnnBenchmarks.AnnDataset,
      config: EvalConfig,
      retrievalLimit: Int = 100
  ): (EvalResult, Double) =
    val inputDim = ann.database.head.length
    val m = math.ceil(config.k / config.activity).toInt

    val trainStart = System.nanoTime()
    val bh = new BioHash(
      BioHashConfig.paper(
        inputDim = inputDim,
        m = m,
        k = config.k,
        learningRate = config.learningRate,
        epochs = config.epochs,
        seed = config.seed,
        normalizeInputs = true
      )
    )
    bh.train(ann.database)
    val trainSeconds = (System.nanoTime() - trainStart) / 1e9

    val dbHashes = bh.encodeAll(ann.database)
    val queryHashes = bh.encodeAll(ann.queries)

    val retrieved = queryHashes.map(q => Retrieval.retrieveTopR(q, dbHashes, retrievalLimit).map(_.index))
    val recall = Metrics.recallAtR(retrieved, ann.groundTruth.map(_.head), retrievalLimit)

    val result = EvalResult(
      method = "BioHash",
      dataset = ann.name,
      k = config.k,
      m = m,
      mAP = recall,
      trainSeconds = trainSeconds,
      encodeSeconds = 0.0,
      querySeconds = 0.0,
      retrievalLimit = retrievalLimit
    )
    (result, recall)

  def formatResult(
      r: EvalResult,
      encodeVectors: Option[Int] = None,
      queryVectors: Option[Int] = None
  ): String =
    val metricName = if r.dataset.contains("sift") || r.dataset.contains("gist") then "recall" else "mAP"
    val throughput = (encodeVectors, queryVectors) match
      case (Some(enc), Some(q)) =>
        val encPerSec =
          if r.encodeSeconds > 0.0 then f" enc/s=${enc / r.encodeSeconds}%.0f" else ""
        val qPerSec =
          if r.querySeconds > 0.0 then f" q/s=${q / r.querySeconds}%.0f" else ""
        encPerSec + qPerSec
      case _ => ""
    f"${r.method}%-14s ${r.dataset}%-16s k=${r.k}%2d m=${r.m}%6d " +
      f"$metricName=${r.mAP}%.4f train=${r.trainSeconds}%.2fs encode=${r.encodeSeconds}%.2fs query=${r.querySeconds}%.2fs" +
      throughput
