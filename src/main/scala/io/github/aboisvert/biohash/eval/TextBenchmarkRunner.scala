// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.*
import io.github.aboisvert.biohash.data.TextBenchmarkDataset

final case class TextTrainResult(
    manifest: TextBenchmarkManifest,
    artifactDir: java.nio.file.Path,
    artifactBytes: Long,
    denseCorpusBytes: Long,
    compressionRatio: Double
)

final case class TextQueryResult(
    method: String,
    dataset: String,
    metrics: TextRetrievalMetrics.MetricSet,
    denseMetrics: Option[TextRetrievalMetrics.MetricSet],
    retrievalLimit: Int,
    queryCount: Int,
    encodeSeconds: Double,
    querySeconds: Double,
    queriesPerSecond: Double,
    artifactBytes: Long,
    denseCorpusBytes: Long,
    compressionRatio: Double
)

object TextBenchmarkRunner:

  def train(
      dataset: TextBenchmarkDataset,
      config: EvalConfig,
      artifactRoot: java.nio.file.Path
  ): TextTrainResult =
    val inputDim = dataset.inputDim
    val m = math.ceil(config.k / config.activity).toInt
    val artifactDir = TextIndexArtifact.artifactDir(artifactRoot, dataset.name, config)

    val trainStart = System.nanoTime()
    val encoder = buildEncoder(config, inputDim, m)
    encoder match
      case bh: BioHash      => bh.train(dataset.corpusVectors)
      case nb: NaiveBioHash => nb.train(dataset.corpusVectors)
      case _: FlyHash       => ()
    val trainSeconds = (System.nanoTime() - trainStart) / 1e9

    val encodeStart = System.nanoTime()
    val corpusHashes = encoder.encodeAll(dataset.corpusVectors)
    val encodeSeconds = (System.nanoTime() - encodeStart) / 1e9

    val manifest = TextBenchmarkManifest(
      dataset = dataset.name,
      method = methodName(config.method),
      inputDim = inputDim,
      k = config.k,
      m = m,
      activity = config.activity,
      epochs = config.epochs,
      learningRate = config.learningRate,
      delta = config.delta,
      antiWinnerRank = config.antiWinnerRank,
      seed = config.seed,
      normalizeInputs = config.normalizeInputs,
      embeddingModel = dataset.embeddingModel,
      corpusSize = dataset.corpusSize,
      querySize = dataset.querySize,
      trainSeconds = trainSeconds,
      encodeSeconds = encodeSeconds,
      createdAt = java.time.Instant.now().toString
    )

    TextIndexArtifact.save(artifactDir, manifest, encoder, dataset.corpusIds, corpusHashes)

    val denseBytes = denseVectorBytes(dataset.corpusVectors)
    val artifactBytes = TextIndexArtifact.directorySizeBytes(artifactDir)
    TextTrainResult(
      manifest = manifest,
      artifactDir = artifactDir,
      artifactBytes = artifactBytes,
      denseCorpusBytes = denseBytes,
      compressionRatio = if artifactBytes > 0 then denseBytes.toDouble / artifactBytes else 0.0
    )

  def query(
      dataset: TextBenchmarkDataset,
      artifactDir: java.nio.file.Path,
      retrievalLimit: Int,
      denseBaseline: Boolean
  ): TextQueryResult =
    val artifact = TextIndexArtifact.load(artifactDir)
    require(artifact.manifest.dataset == dataset.name, "Artifact dataset does not match query dataset")

    val encodeStart = System.nanoTime()
    val queryHashes = artifact.encoder.encodeAll(dataset.queryVectors)
    val encodeSeconds = (System.nanoTime() - encodeStart) / 1e9

    val queryStart = System.nanoTime()
    val retrievedIndices = queryHashes.map { q =>
      Retrieval.retrieveTopR(q, artifact.corpusHashes, retrievalLimit).map(_.index)
    }
    val retrievedDocIds = retrievedIndices.map { indices =>
      indices.map(artifact.corpusIds(_))
    }
    val queryIdsWithQrels = dataset.queryIds.filter(dataset.qrels.contains)
    val filteredRetrieved = dataset.queryIds
      .zip(retrievedDocIds)
      .collect { case (qid, docs) if dataset.qrels.contains(qid) => docs }
      .toIndexedSeq
    val metrics = TextRetrievalMetrics.evaluate(filteredRetrieved, queryIdsWithQrels, dataset.qrels)
    val querySeconds = (System.nanoTime() - queryStart) / 1e9

    val denseMetrics =
      if denseBaseline && dataset.corpusSize <= 250000 then Some(runDenseBaseline(dataset, retrievalLimit))
      else None

    val denseBytes = denseVectorBytes(dataset.corpusVectors)
    val artifactBytes = TextIndexArtifact.directorySizeBytes(artifactDir)

    TextQueryResult(
      method = artifact.manifest.method,
      dataset = dataset.name,
      metrics = metrics,
      denseMetrics = denseMetrics,
      retrievalLimit = retrievalLimit,
      queryCount = queryIdsWithQrels.length,
      encodeSeconds = encodeSeconds,
      querySeconds = querySeconds,
      queriesPerSecond = if querySeconds > 0 then queryIdsWithQrels.length / querySeconds else 0.0,
      artifactBytes = artifactBytes,
      denseCorpusBytes = denseBytes,
      compressionRatio = if artifactBytes > 0 then denseBytes.toDouble / artifactBytes else 0.0
    )

  def formatTrainResult(result: TextTrainResult): String =
    val m = result.manifest
    f"trained ${m.method}%-14s dataset=${m.dataset}%-12s k=${m.k}%2d m=${m.m}%6d " +
      f"corpus=${m.corpusSize}%6d train=${m.trainSeconds}%.2fs encode=${m.encodeSeconds}%.2fs " +
      f"artifact=${result.artifactDir} size=${result.artifactBytes / 1024}%.0fKB " +
      f"compression=${result.compressionRatio}%.1fx"

  def formatQueryResult(result: TextQueryResult): String =
    val lines = scala.collection.mutable.ArrayBuffer.empty[String]
    lines += f"${result.method}%-14s dataset=${result.dataset}%-12s queries=${result.queryCount}%4d R=${result.retrievalLimit}%3d " +
      TextRetrievalMetrics.formatMetrics(result.metrics)
    lines += f"encode=${result.encodeSeconds}%.2fs query=${result.querySeconds}%.2fs q/s=${result.queriesPerSecond}%.0f " +
      f"artifact=${result.artifactBytes / 1024}%.0fKB dense=${result.denseCorpusBytes / 1024}%.0fKB compression=${result.compressionRatio}%.1fx"
    result.denseMetrics.foreach { dense =>
      lines += "dense-baseline " + TextRetrievalMetrics.formatMetrics(dense)
    }
    lines.mkString("\n")

  private def buildEncoder(config: EvalConfig, inputDim: Int, m: Int): HashEncoder =
    config.method match
      case HashMethod.BioHash =>
        new BioHash(
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
      case HashMethod.FlyHash =>
        new FlyHash(
          FlyHashConfig(
            inputDim = inputDim,
            m = m,
            k = config.k,
            samplingRate = 0.1,
            seed = config.seed,
            normalizeInputs = config.normalizeInputs
          )
        )
      case HashMethod.NaiveBioHash =>
        new NaiveBioHash(
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

  private def methodName(method: HashMethod): String =
    method match
      case HashMethod.BioHash      => "BioHash"
      case HashMethod.FlyHash      => "FlyHash"
      case HashMethod.NaiveBioHash => "NaiveBioHash"

  private def denseVectorBytes(vectors: IndexedSeq[Array[Double]]): Long =
    vectors.length.toLong * (if vectors.isEmpty then 0 else vectors.head.length) * 8L

  private def runDenseBaseline(dataset: TextBenchmarkDataset, retrievalLimit: Int): TextRetrievalMetrics.MetricSet =
    val corpus = dataset.corpusVectors
    val normalizedCorpus = corpus.map(VectorOps.l2NormalizeInput)
    val queryIdsWithQrels = dataset.queryIds.filter(dataset.qrels.contains)
    val retrievedDocIds = dataset.queryIds
      .zip(dataset.queryVectors)
      .collect {
        case (qid, vector) if dataset.qrels.contains(qid) =>
          val query = VectorOps.l2NormalizeInput(vector)
          normalizedCorpus.zipWithIndex
            .map { case (doc, idx) => (VectorOps.dot(query, doc), idx) }
            .sortBy { case (score, idx) => (-score, idx) }
            .take(retrievalLimit)
            .map { case (_, idx) => dataset.corpusIds(idx) }
      }
      .toIndexedSeq
    TextRetrievalMetrics.evaluate(retrievedDocIds, queryIdsWithQrels, dataset.qrels)
