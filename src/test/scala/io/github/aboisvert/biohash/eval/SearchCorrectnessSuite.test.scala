// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.*
import io.github.aboisvert.biohash.testkit.RetrievalOracle
import java.nio.file.Files

class SearchCorrectnessSuite extends munit.FunSuite:

  private def sampleManifest(corpusSize: Int): TextBenchmarkManifest =
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
      corpusSize = corpusSize,
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
    IndexedSeq(
      SparseHash(Array(0, 1), 2),
      SparseHash(Array(1, 2), 2),
      SparseHash(Array(0, 2), 2)
    )

  private def sampleIds: IndexedSeq[String] =
    IndexedSeq("d1", "d2", "d3")

  private def assertSearchMatchesOracle(
      service: TextSearchService,
      segment: TextIndexSegment,
      queryVector: Array[Double],
      k: Int
  ): Unit =
    val queryHash = segment.encoder.encode(queryVector)
    val oracle = RetrievalOracle.retrieveTopR(queryHash, segment.corpusHashes, k)
    val hits = service.search(queryVector, k)
    assertEquals(hits.length, oracle.length)
    hits.zip(oracle).zipWithIndex.foreach { case ((hit, expected), rank) =>
      assertEquals(hit.rank, rank + 1, s"rank at position $rank")
      assertEquals(hit.docId, segment.corpusIds(expected.index), s"docId at position $rank")
      assertEquals(hit.hamming, expected.distance, s"hamming at position $rank")
    }

  test("single-segment TextSearchService matches retrieval oracle") {
    val dir = Files.createTempDirectory("biohash-search-single")
    val ids = sampleIds
    TextIndexArtifact.save(dir, sampleManifest(ids.length), sampleEncoder, ids, sampleHashes)
    val service = TextSearchService.load(dir)
    val artifact = TextIndexArtifact.load(dir)
    assertEquals(artifact.segmentCount, 1)
    assertSearchMatchesOracle(service, artifact.latestSegment, Array(1.0, 0.0, 0.0), k = 2)
    assertSearchMatchesOracle(service, artifact.latestSegment, Array(0.0, 1.0, 0.5), k = 3)
  }

  test("multi-segment segmentCandidates match per-segment oracle") {
    val dir = Files.createTempDirectory("biohash-search-multi")
    TextIndexArtifact.save(dir, sampleManifest(2), sampleEncoder, IndexedSeq("d1", "d2"), sampleHashes.take(2))
    TextIndexArtifact.appendSegment(
      dir,
      sampleEncoder,
      IndexedSeq("d3"),
      IndexedSeq(sampleHashes(2)),
      trainingSteps = 3L
    )
    val artifact = TextIndexArtifact.load(dir)
    assertEquals(artifact.segmentCount, 2)
    val query = Array(1.0, 0.0, 0.0)
    artifact.segments.foreach { segment =>
      val queryHash = segment.encoder.encode(query)
      val oracle = RetrievalOracle.retrieveTopR(queryHash, segment.corpusHashes, segment.corpusHashes.length)
      val candidates = SegmentedRetrieval.segmentCandidates(query, segment, normalizeCrossSegmentDistances = false)
      assertEquals(candidates.length, oracle.length, s"segment ${segment.segmentId}")
      candidates.zip(oracle).foreach { case (candidate, expected) =>
        assertEquals(candidate.segmentId, segment.segmentId)
        assertEquals(candidate.corpusId, segment.corpusIds(expected.index))
        assertEquals(candidate.distance, expected.distance)
      }
    }
  }

  test("consolidated artifact search matches oracle") {
    val dir = Files.createTempDirectory("biohash-search-consolidated")
    TextIndexArtifact.save(dir, sampleManifest(2), sampleEncoder, IndexedSeq("d1", "d2"), sampleHashes.take(2))
    TextIndexArtifact.appendSegment(
      dir,
      sampleEncoder,
      IndexedSeq("d3"),
      IndexedSeq(sampleHashes(2)),
      trainingSteps = 3L
    )
    val consolidatedIds = sampleIds
    val consolidatedHashes = sampleHashes
    TextIndexArtifact.replaceWithSingleSegment(dir, sampleEncoder, consolidatedIds, consolidatedHashes, 3L)

    val artifact = TextIndexArtifact.load(dir)
    assertEquals(artifact.segmentCount, 1)
    val service = TextSearchService.load(dir)
    assertSearchMatchesOracle(service, artifact.latestSegment, Array(1.0, 0.0, 0.0), k = 3)
  }
