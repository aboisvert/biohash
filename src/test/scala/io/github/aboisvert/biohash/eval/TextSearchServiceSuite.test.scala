// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.data.{TextBenchmark, TextBenchmarkFixtures}
import java.nio.file.Files
import scala.compiletime.uninitialized

class TextSearchServiceSuite extends munit.FunSuite:

  private var fixtureDir: java.nio.file.Path = uninitialized
  private var artifactDir: java.nio.file.Path = uninitialized

  override def beforeAll(): Unit =
    fixtureDir = TextBenchmarkFixtures.writeFixture(Files.createTempDirectory("biohash-text-search-fixture"))
    val dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val config = EvalConfig(k = 2, activity = 0.5, epochs = 2, seed = 7L, normalizeInputs = true)
    val artifactRoot = Files.createTempDirectory("biohash-text-search-artifacts")
    val trained = TextBenchmarkRunner.train(dataset, config, artifactRoot)
    artifactDir = trained.artifactDir

  test("loadCorpusPassages reads title and body") {
    val passages = TextBenchmark.loadCorpusPassages(fixtureDir)
    assertEquals(passages("d1"), "A\nalpha")
    assertEquals(passages("d3"), "C\nbeta")
  }

  test("loadQueryTexts and listQueryIds") {
    val queries = TextBenchmark.loadQueryTexts(fixtureDir)
    assertEquals(queries("q1"), "find alpha topic")
    assertEquals(TextBenchmark.listQueryIds(fixtureDir, 10), IndexedSeq("q1", "q2"))
  }

  test("search returns ranked doc ids for a benchmark query vector") {
    val dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val service = TextSearchService.load(artifactDir)
    val q1Vector = dataset.queryVectors.head
    val hits = service.search(q1Vector, k = 2)
    assertEquals(hits.length, 2)
    assertEquals(hits.head.rank, 1)
    assertEquals(hits.head.docId, "d1")
    assert(hits.forall(_.hamming >= 0))
  }

  test("search rejects wrong vector dimension") {
    val service = TextSearchService.load(artifactDir)
    intercept[IllegalArgumentException](service.search(Array(1.0, 0.0), k = 1))
  }
