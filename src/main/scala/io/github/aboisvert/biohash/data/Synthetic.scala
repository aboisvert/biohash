// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.data

import io.github.aboisvert.biohash.VectorOps
import scala.util.Random

/** Knobs for synthetic LLM-style passage retrieval (clustered embedding vectors). */
final case class TextRetrievalConfig(
    corpusSize: Int = 50000,
    queryCount: Int = 1000,
    dim: Int = 768,
    topics: Int = 500,
    clustersPerTopic: Int = 4,
    noise: Double = 0.15,
    hardNegativeRate: Double = 0.10,
    seed: Long = 42L
):

  require(corpusSize > 0, "corpusSize must be positive")
  require(queryCount > 0, "queryCount must be positive")
  require(dim > 0, "dim must be positive")
  require(topics > 0, "topics must be positive")
  require(clustersPerTopic > 0, "clustersPerTopic must be positive")
  require(noise >= 0.0, "noise must be non-negative")
  require(hardNegativeRate >= 0.0 && hardNegativeRate <= 1.0, "hardNegativeRate must be in [0, 1]")

object Synthetic:

  /** Two-class blob dataset for fast sanity checks. */
  def twoClassBlobs(
      dim: Int = 8,
      perClass: Int = 50,
      seed: Long = 0L
  ): LabeledDataset =
    val rng = Random(seed)
    val class0 = (0 until perClass).map { _ =>
      Array.tabulate(dim)(i => if i < dim / 2 then rng.nextGaussian() + 2.0 else rng.nextGaussian())
    }
    val class1 = (0 until perClass).map { _ =>
      Array.tabulate(dim)(i => if i < dim / 2 then rng.nextGaussian() else rng.nextGaussian() + 2.0)
    }
    val vectors = class0 ++ class1
    val labels = IndexedSeq.fill(perClass)(0) ++ IndexedSeq.fill(perClass)(1)
    LabeledDataset(vectors.toIndexedSeq, labels, "two-class-blobs")

  def randomUnitVectors(n: Int, dim: Int, seed: Long = 0L): IndexedSeq[Array[Double]] =
    val rng = Random(seed)
    (0 until n).map { _ =>
      val v = Array.fill(dim)(rng.nextGaussian())
      VectorOps.normalizeInPlace(v, 2.0)
      v
    }.toIndexedSeq

  /** Synthetic passage/query retrieval split inspired by LLM embedding search.
    *
    * Passages are clustered by topic with optional subclusters; queries are noisy
    * paraphrases of randomly chosen passages. Relevance is defined by shared topic id.
    */
  def textRetrievalSplit(config: TextRetrievalConfig): RetrievalSplit =
    val centroidRng = Random(config.seed)
    val passageRng = Random(config.seed + 1L)
    val queryRng = Random(config.seed + 2L)

    val topicCentroids = Array.tabulate(config.topics) { _ =>
      val c = Array.fill(config.dim)(centroidRng.nextGaussian())
      VectorOps.normalizeInPlace(c, 2.0)
      c
    }

    val clusterOffsets = Array.tabulate(config.topics) { topic =>
      Array.tabulate(config.clustersPerTopic) { _ =>
        Array.fill(config.dim)(centroidRng.nextGaussian() * 0.2)
      }
    }

    val passages = Array.ofDim[Array[Double]](config.corpusSize)
    val passageLabels = Array.ofDim[Int](config.corpusSize)

    var i = 0
    while i < config.corpusSize do
      val topic = passageRng.nextInt(config.topics)
      val cluster = passageRng.nextInt(config.clustersPerTopic)
      val v = Array.fill(config.dim)(0.0)
      addScaled(v, topicCentroids(topic), 1.0)
      addScaled(v, clusterOffsets(topic)(cluster), 1.0)
      if config.hardNegativeRate > 0.0 && config.topics > 1 && passageRng.nextDouble() < config.hardNegativeRate then
        val neighbor = (topic + 1 + passageRng.nextInt(config.topics - 1)) % config.topics
        addScaled(v, topicCentroids(neighbor), config.noise * 0.5)
      addGaussianNoise(v, passageRng, config.noise)
      VectorOps.normalizeInPlace(v, 2.0)
      passages(i) = v
      passageLabels(i) = topic
      i += 1

    val queries = Array.ofDim[Array[Double]](config.queryCount)
    val queryLabels = Array.ofDim[Int](config.queryCount)

    var q = 0
    while q < config.queryCount do
      val targetIdx = queryRng.nextInt(config.corpusSize)
      val topic = passageLabels(targetIdx)
      val v = passages(targetIdx).clone()
      addGaussianNoise(v, queryRng, config.noise * 1.5)
      VectorOps.normalizeInPlace(v, 2.0)
      queries(q) = v
      queryLabels(q) = topic
      q += 1

    val dbVectors = passages.toIndexedSeq
    val dbLabels = passageLabels.toIndexedSeq
    RetrievalSplit(
      trainVectors = dbVectors,
      databaseVectors = dbVectors,
      databaseLabels = dbLabels,
      queryVectors = queries.toIndexedSeq,
      queryLabels = queryLabels.toIndexedSeq
    )

  private def addScaled(target: Array[Double], source: Array[Double], scale: Double): Unit =
    var i = 0
    while i < target.length do
      target(i) += source(i) * scale
      i += 1

  private def addGaussianNoise(v: Array[Double], rng: Random, sigma: Double): Unit =
    if sigma > 0.0 then
      var i = 0
      while i < v.length do
        v(i) += rng.nextGaussian() * sigma
        i += 1
