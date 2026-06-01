// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.data

import io.github.aboisvert.biohash.eval.*
import java.nio.file.Files
import scala.compiletime.uninitialized

class TextBenchmarkSuite extends munit.FunSuite:

  private var fixtureDir: java.nio.file.Path = uninitialized

  override def beforeAll(): Unit =
    fixtureDir = TextBenchmarkFixtures.writeFixture(Files.createTempDirectory("biohash-text-fixture"))

  test("isAvailable detects fixture layout") {
    assert(TextBenchmark.isAvailable(fixtureDir))
  }

  test("load reads ids, vectors, and qrels in order") {
    val dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    assertEquals(dataset.corpusIds, IndexedSeq("d1", "d2", "d3", "d4"))
    assertEquals(dataset.queryIds, IndexedSeq("q1", "q2"))
    assertEquals(dataset.corpusSize, 4)
    assertEquals(dataset.querySize, 2)
    assertEquals(dataset.inputDim, 3)
    assertEquals(dataset.embeddingModel, "test-fixture")
    assertEquals(dataset.qrels("q1")("d1"), 2)
    assertEquals(dataset.qrels("q2")("d3"), 1)
  }

  test("parseJsonlId extracts BEIR ids") {
    val id = TextBenchmark.parseJsonlId("""{"_id":"doc-1","title":"T","text":"body"}""")
    assertEquals(id, Some("doc-1"))
  }

  test("train and query benchmark round-trip on fixture") {
    val dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val config = EvalConfig(k = 2, activity = 0.5, epochs = 2, seed = 7L, normalizeInputs = true)
    val artifactRoot = Files.createTempDirectory("biohash-text-artifacts")
    val trainResult = TextBenchmarkRunner.train(dataset, config, artifactRoot)
    assert(trainResult.manifest.corpusSize == 4)
    assert(Files.exists(trainResult.artifactDir.resolve(TextIndexArtifact.ManifestFile)))

    val queryResult = TextBenchmarkRunner.query(
      dataset = dataset,
      artifactDir = trainResult.artifactDir,
      retrievalLimit = 3,
      denseBaseline = true
    )
    assertEquals(queryResult.queryCount, 2)
    assert(queryResult.metrics.ndcgAt10 >= 0.0)
    assert(queryResult.denseMetrics.isDefined)
    assert(queryResult.compressionRatio > 0.0)
  }

  test("artifact save/load preserves corpus hashes") {
    val dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val config = EvalConfig(k = 2, activity = 0.5, epochs = 1, seed = 3L, normalizeInputs = true, method = HashMethod.FlyHash)
    val artifactRoot = Files.createTempDirectory("biohash-text-artifacts-fly")
    val trained = TextBenchmarkRunner.train(dataset, config, artifactRoot)
    val loaded = TextIndexArtifact.load(trained.artifactDir)
    assertEquals(loaded.corpusIds, dataset.corpusIds)
    assertEquals(loaded.corpusHashes.length, dataset.corpusSize)
    loaded.corpusHashes.foreach(hash => assertEquals(hash.k, 2))
  }
