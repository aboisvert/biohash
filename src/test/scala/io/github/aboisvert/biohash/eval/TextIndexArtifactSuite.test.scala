// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.*
import java.nio.file.Files

class TextIndexArtifactSuite extends munit.FunSuite:

  private def sampleManifest: TextBenchmarkManifest =
    TextBenchmarkManifest(
      dataset = "mini",
      method = "BioHash",
      inputDim = 3,
      k = 2,
      m = 4,
      activity = 0.5,
      epochs = 1,
      learningRate = 0.01,
      delta = 0.0,
      antiWinnerRank = 2,
      seed = 1L,
      normalizeInputs = true,
      embeddingModel = "test",
      corpusSize = 2,
      querySize = 1,
      trainSeconds = 0.1,
      encodeSeconds = 0.1,
      createdAt = "2026-01-01T00:00:00Z",
      segmentCount = 1,
      totalTrainingSteps = 2L
    )

  private def sampleEncoder: BioHash =
    BioHash.fromWeights(
      BioHashConfig.paper(inputDim = 3, m = 4, k = 2, seed = 1L, normalizeInputs = true),
      Array(
        Array(1.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0),
        Array(0.0, 0.0, 1.0),
        Array(1.0, 1.0, 0.0)
      ),
      trainingSteps = 2L
    )

  private def sampleHashes: IndexedSeq[SparseHash] =
    IndexedSeq(SparseHash(Array(0, 1), 2), SparseHash(Array(1, 2), 2))

  test("legacy artifact loads as single segment 0") {
    val dir = Files.createTempDirectory("biohash-artifact-legacy")
    val ids = IndexedSeq("d1", "d2")
    TextIndexArtifact.save(dir, sampleManifest, sampleEncoder, ids, sampleHashes)
    assert(!Files.exists(dir.resolve(TextIndexArtifact.SegmentsFile)))

    val loaded = TextIndexArtifact.load(dir)
    assertEquals(loaded.segmentCount, 1)
    assertEquals(loaded.segments.head.segmentId, 0)
    assertEquals(loaded.corpusIds, ids)
    assertEquals(loaded.corpusHashes.map(_.active.toSeq), sampleHashes.map(_.active.toSeq))
  }

  test("extendLatestSegment preserves existing ids and hashes") {
    val dir = Files.createTempDirectory("biohash-artifact-extend")
    val ids = IndexedSeq("d1", "d2")
    TextIndexArtifact.save(dir, sampleManifest, sampleEncoder, ids, sampleHashes)
    val before = TextIndexArtifact.load(dir)

    val newIds = IndexedSeq("d3")
    val newHashes = IndexedSeq(SparseHash(Array(2, 3), 2))
    TextIndexArtifact.extendLatestSegment(dir, newIds, newHashes)

    val after = TextIndexArtifact.load(dir)
    assertEquals(after.corpusIds.take(2), before.corpusIds)
    assertEquals(after.corpusHashes.take(2).map(_.active.toSeq), before.corpusHashes.map(_.active.toSeq))
    assertEquals(after.corpusIds, IndexedSeq("d1", "d2", "d3"))
    assertEquals(after.manifest.corpusSize, 3)
  }

  test("appendSegment adds a new segment and segments index file") {
    val dir = Files.createTempDirectory("biohash-artifact-append")
    val ids = IndexedSeq("d1", "d2")
    TextIndexArtifact.save(dir, sampleManifest, sampleEncoder, ids, sampleHashes)

    val updatedEncoder = BioHash.fromWeights(
      BioHashConfig.paper(inputDim = 3, m = 4, k = 2, seed = 1L, normalizeInputs = true),
      sampleEncoder.weights,
      trainingSteps = 4L
    )
    val newIds = IndexedSeq("d3")
    val newHashes = IndexedSeq(SparseHash(Array(0, 2), 2))
    TextIndexArtifact.appendSegment(dir, updatedEncoder, newIds, newHashes, trainingSteps = 4L)

    assert(Files.exists(dir.resolve(TextIndexArtifact.SegmentsFile)))
    val loaded = TextIndexArtifact.load(dir)
    assertEquals(loaded.segmentCount, 2)
    assertEquals(loaded.corpusIds, IndexedSeq("d1", "d2", "d3"))
    assertEquals(loaded.segments.head.corpusIds, IndexedSeq("d1", "d2"))
    assertEquals(loaded.segments(1).corpusIds, IndexedSeq("d3"))
  }

  test("replaceWithSingleSegment collapses to one segment") {
    val dir = Files.createTempDirectory("biohash-artifact-consolidate")
    val ids = IndexedSeq("d1", "d2")
    TextIndexArtifact.save(dir, sampleManifest, sampleEncoder, ids, sampleHashes)
    val updatedEncoder = sampleEncoder
    TextIndexArtifact.appendSegment(dir, updatedEncoder, IndexedSeq("d3"), IndexedSeq(SparseHash(Array(1, 3), 2)), 3L)
    assertEquals(TextIndexArtifact.load(dir).segmentCount, 2)

    val consolidatedIds = IndexedSeq("d1", "d2", "d3")
    val consolidatedHashes =
      IndexedSeq(SparseHash(Array(0, 1), 2), SparseHash(Array(1, 2), 2), SparseHash(Array(2, 3), 2))
    TextIndexArtifact.replaceWithSingleSegment(dir, updatedEncoder, consolidatedIds, consolidatedHashes, 3L)

    val loaded = TextIndexArtifact.load(dir)
    assertEquals(loaded.segmentCount, 1)
    assertEquals(loaded.manifest.segmentCount, 1)
    assertEquals(loaded.corpusIds, consolidatedIds)
    assert(!Files.exists(dir.resolve(TextIndexArtifact.SegmentsFile)))
    assert(!Files.exists(dir.resolve("encoder-1.bin")))
  }

  test("segmented retrieval merges deterministically across segments") {
    val dir = Files.createTempDirectory("biohash-artifact-query")
    val ids = IndexedSeq("d1", "d2")
    TextIndexArtifact.save(dir, sampleManifest, sampleEncoder, ids, sampleHashes)
    TextIndexArtifact.appendSegment(
      dir,
      sampleEncoder,
      IndexedSeq("d3"),
      IndexedSeq(SparseHash(Array(0, 2), 2)),
      trainingSteps = 3L
    )
    val artifact = TextIndexArtifact.load(dir)
    val query = Array(1.0, 0.0, 0.0)
    val first = SegmentedRetrieval.retrieveTopR(query, artifact.segments, r = 2)
    val second = SegmentedRetrieval.retrieveTopR(query, artifact.segments, r = 2)
    assertEquals(first, second)
    assertEquals(first.length, 2)
  }
