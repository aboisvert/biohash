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

final case class TextAppendResult(
    artifactDir: java.nio.file.Path,
    addedCount: Int,
    corpusSize: Int,
    segmentCount: Int
)

final case class TextImproveResult(
    artifactDir: java.nio.file.Path,
    addedCount: Int,
    corpusSize: Int,
    segmentCount: Int,
    totalTrainingSteps: Long
)

final case class TextConsolidateResult(
    artifactDir: java.nio.file.Path,
    corpusSize: Int,
    segmentCount: Int,
    totalTrainingSteps: Long
)

final case class IncrementalUpdateConfig(
    epochs: Int = 1,
    learningRate: Option[Double] = None
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
    compressionRatio: Double,
    segmentCount: Int = 1
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
    val totalTrainingSteps = encoder match
      case bh: BioHash =>
        bh.train(dataset.corpusVectors)
        bh.currentTrainingSteps
      case nb: NaiveBioHash =>
        nb.train(dataset.corpusVectors)
        nb.currentTrainingSteps
      case _: FlyHash => 0L
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
      createdAt = java.time.Instant.now().toString,
      segmentCount = 1,
      totalTrainingSteps = totalTrainingSteps
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

  /** Append new items using the latest segment's frozen encoder (exact within that segment). */
  def appendItems(
      artifactDir: java.nio.file.Path,
      newIds: IndexedSeq[String],
      newVectors: IndexedSeq[Array[Double]]
  ): TextAppendResult =
    require(newIds.length == newVectors.length, "appendItems: ids and vectors length mismatch")
    val artifact = TextIndexArtifact.load(artifactDir)
    val encoder = artifact.latestSegment.encoder
    newVectors.foreach(v =>
      require(v.length == artifact.manifest.inputDim, s"appendItems: expected dim ${artifact.manifest.inputDim}")
    )
    val newHashes = encoder.encodeAll(newVectors)
    val updated = TextIndexArtifact.extendLatestSegment(artifactDir, newIds, newHashes)
    TextAppendResult(
      artifactDir = artifactDir,
      addedCount = newIds.length,
      corpusSize = updated.manifest.corpusSize,
      segmentCount = updated.segmentCount
    )

  /** Train the latest encoder on new vectors and store them in a new segment. */
  def improveEncoder(
      artifactDir: java.nio.file.Path,
      newIds: IndexedSeq[String],
      newVectors: IndexedSeq[Array[Double]],
      updateConfig: IncrementalUpdateConfig = IncrementalUpdateConfig()
  ): TextImproveResult =
    require(newIds.length == newVectors.length, "improveEncoder: ids and vectors length mismatch")
    val artifact = TextIndexArtifact.load(artifactDir)
    require(
      !artifact.manifest.method.equalsIgnoreCase("flyhash"),
      "improveEncoder: FlyHash has no training step"
    )
    newVectors.foreach(v =>
      require(v.length == artifact.manifest.inputDim, s"improveEncoder: expected dim ${artifact.manifest.inputDim}")
    )
    val latest = artifact.latestSegment
    val trainedEncoder = trainLatestEncoder(latest, artifact.manifest, newVectors, updateConfig)
    val newHashes = trainedEncoder.encodeAll(newVectors)
    val trainingSteps = trainedEncoder match
      case bh: BioHash      => bh.currentTrainingSteps
      case nb: NaiveBioHash => nb.currentTrainingSteps
      case other            => throw IllegalStateException(s"Unexpected trainable encoder: $other")
    val updated = TextIndexArtifact.appendSegment(
      artifactDir,
      trainedEncoder,
      newIds,
      newHashes,
      trainingSteps
    )
    TextImproveResult(
      artifactDir = artifactDir,
      addedCount = newIds.length,
      corpusSize = updated.manifest.corpusSize,
      segmentCount = updated.segmentCount,
      totalTrainingSteps = trainingSteps
    )

  /** Re-encode all indexed items with the newest encoder into a single segment. */
  def consolidate(
      artifactDir: java.nio.file.Path,
      vectorsById: Map[String, Array[Double]]
  ): TextConsolidateResult =
    val artifact = TextIndexArtifact.load(artifactDir)
    require(
      !artifact.manifest.method.equalsIgnoreCase("flyhash"),
      "consolidate: FlyHash consolidation is not supported"
    )
    val latestEncoder = artifact.latestSegment.encoder
    val allIds = artifact.corpusIds
    allIds.foreach { id =>
      require(vectorsById.contains(id), s"consolidate: missing dense vector for corpus id $id")
    }
    val allVectors = allIds.map(vectorsById(_))
    val allHashes = latestEncoder.encodeAll(allVectors)
    val trainingSteps = latestEncoder match
      case bh: BioHash      => bh.currentTrainingSteps
      case nb: NaiveBioHash => nb.currentTrainingSteps
      case other            => throw IllegalStateException(s"Unexpected encoder for consolidation: $other")
    val updated =
      TextIndexArtifact.replaceWithSingleSegment(artifactDir, latestEncoder, allIds, allHashes, trainingSteps)
    TextConsolidateResult(
      artifactDir = artifactDir,
      corpusSize = updated.manifest.corpusSize,
      segmentCount = updated.segmentCount,
      totalTrainingSteps = trainingSteps
    )

  def query(
      dataset: TextBenchmarkDataset,
      artifactDir: java.nio.file.Path,
      retrievalLimit: Int,
      denseBaseline: Boolean,
      normalizeCrossSegmentDistances: Boolean = false
  ): TextQueryResult =
    val artifact = TextIndexArtifact.load(artifactDir)
    require(artifact.manifest.dataset == dataset.name, "Artifact dataset does not match query dataset")

    val encodeStart = System.nanoTime()
    val retrievedDocIds =
      if artifact.segmentCount == 1 then
        val queryHashes = artifact.latestSegment.encoder.encodeAll(dataset.queryVectors)
        val encodeSeconds = (System.nanoTime() - encodeStart) / 1e9
        val queryStart = System.nanoTime()
        val docIds = queryHashes.map { queryHash =>
          Retrieval
            .retrieveTopR(queryHash, artifact.latestSegment.corpusHashes, retrievalLimit)
            .map(result => artifact.latestSegment.corpusIds(result.index))
        }
        val querySeconds = (System.nanoTime() - queryStart) / 1e9
        (docIds, encodeSeconds, querySeconds)
      else
        val queryStart = System.nanoTime()
        val docIds = dataset.queryVectors.map { queryVector =>
          SegmentedRetrieval.retrieveTopR(
            queryVector,
            artifact.segments,
            retrievalLimit,
            normalizeCrossSegmentDistances
          )
        }
        val totalQuerySeconds = (System.nanoTime() - queryStart) / 1e9
        // Multi-segment path encodes once per segment inside retrieval; attribute all time to query.
        (docIds, 0.0, totalQuerySeconds)

    val queryIdsWithQrels = dataset.queryIds.filter(dataset.qrels.contains)
    val filteredRetrieved = dataset.queryIds
      .zip(retrievedDocIds._1)
      .collect { case (qid, docs) if dataset.qrels.contains(qid) => docs }
      .toIndexedSeq
    val metrics = TextRetrievalMetrics.evaluate(filteredRetrieved, queryIdsWithQrels, dataset.qrels)

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
      encodeSeconds = retrievedDocIds._2,
      querySeconds = retrievedDocIds._3,
      queriesPerSecond =
        if retrievedDocIds._3 > 0 then queryIdsWithQrels.length / retrievedDocIds._3 else 0.0,
      artifactBytes = artifactBytes,
      denseCorpusBytes = denseBytes,
      compressionRatio = if artifactBytes > 0 then denseBytes.toDouble / artifactBytes else 0.0,
      segmentCount = artifact.segmentCount
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
      f"segments=${result.segmentCount}%2d artifact=${result.artifactBytes / 1024}%.0fKB dense=${result.denseCorpusBytes / 1024}%.0fKB compression=${result.compressionRatio}%.1fx"
    result.denseMetrics.foreach { dense =>
      lines += "dense-baseline " + TextRetrievalMetrics.formatMetrics(dense)
    }
    lines.mkString("\n")

  private def trainLatestEncoder(
      latestSegment: TextIndexSegment,
      manifest: TextBenchmarkManifest,
      newVectors: IndexedSeq[Array[Double]],
      updateConfig: IncrementalUpdateConfig
  ): HashEncoder =
    val learningRate = updateConfig.learningRate.getOrElse(manifest.learningRate)
    val shuffleSeedOffset = latestSegment.manifest.trainingSteps
    latestSegment.encoder match
      case bh: BioHash =>
        val config = BioHashConfig.paper(
          inputDim = manifest.inputDim,
          m = manifest.m,
          k = manifest.k,
          learningRate = learningRate,
          epochs = manifest.epochs,
          antiWinnerRank = manifest.antiWinnerRank,
          delta = manifest.delta,
          seed = manifest.seed,
          normalizeInputs = manifest.normalizeInputs
        )
        val restored = BioHash.fromWeights(config, bh.weights, latestSegment.manifest.trainingSteps)
        restored.trainMiniBatch(newVectors, epochs = updateConfig.epochs, shuffleSeedOffset = shuffleSeedOffset)
        restored
      case nb: NaiveBioHash =>
        val config = NaiveBioHashConfig.paper(
          inputDim = manifest.inputDim,
          k = manifest.k,
          learningRate = learningRate,
          epochs = manifest.epochs,
          antiWinnerRank = manifest.antiWinnerRank,
          delta = manifest.delta,
          seed = manifest.seed,
          normalizeInputs = manifest.normalizeInputs
        )
        val restored = NaiveBioHash.fromWeights(config, nb.savedWeights, latestSegment.manifest.trainingSteps)
        restored.trainMiniBatch(newVectors, epochs = updateConfig.epochs, shuffleSeedOffset = shuffleSeedOffset)
        restored
      case other =>
        throw IllegalArgumentException(s"improveEncoder: unsupported encoder type $other")

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
