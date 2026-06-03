// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.testkit.{HashGenerators, RetrievalOracle}
import scala.util.Random

class RetrievalCorrectnessSuite extends munit.FunSuite:

  private def assertResultsEqual(
      actual: IndexedSeq[RetrievalResult],
      expected: IndexedSeq[RetrievalResult],
      context: String
  ): Unit =
    assertEquals(actual.map(r => (r.index, r.distance)), expected.map(r => (r.index, r.distance)), context)

  test("retrieveTopR matches oracle over random inputs") {
    val rng = Random(42L)
    var iteration = 0
    while iteration < 100 do
      val k = rng.nextInt(8) + 1
      val m = k + rng.nextInt(32)
      val n = rng.nextInt(50) + 1
      val db = HashGenerators.randomDatabase(rng, n, k, m)
      val query = HashGenerators.randomSparseHash(rng, k, m)
      val r = rng.nextInt(n + 5) + 1
      val exclude = HashGenerators.randomExclude(rng, n)
      val expected = RetrievalOracle.retrieveTopR(query, db, r, exclude)
      val actual = Retrieval.retrieveTopR(query, db, r, exclude)
      assertResultsEqual(actual, expected, s"iteration=$iteration k=$k m=$m n=$n r=$r")
      iteration += 1
  }

  test("retrieveAll matches oracle") {
    val rng = Random(99L)
    var iteration = 0
    while iteration < 50 do
      val k = rng.nextInt(6) + 1
      val m = k + rng.nextInt(24)
      val n = rng.nextInt(40) + 1
      val db = HashGenerators.randomDatabase(rng, n, k, m)
      val query = HashGenerators.randomSparseHash(rng, k, m)
      val exclude = HashGenerators.randomExclude(rng, n)
      val expected = RetrievalOracle.retrieveTopR(query, db, n, exclude)
      val actual = Retrieval.retrieveAll(query, db, exclude)
      assertResultsEqual(actual, expected, s"iteration=$iteration")
      iteration += 1
  }

  test("batchRetrieveTopR matches sequential retrieveTopR") {
    val rng = Random(7L)
    val k = 4
    val m = 16
    val n = 30
    val db = HashGenerators.randomDatabase(rng, n, k, m)
    val queries = IndexedSeq.tabulate(40)(_ => HashGenerators.randomSparseHash(rng, k, m))
    val r = 5
    val batch = Retrieval.batchRetrieveTopR(queries, db, r)
    assertEquals(batch.length, queries.length)
    queries.zip(batch).zipWithIndex.foreach { case ((query, results), idx) =>
      val expected = RetrievalOracle.retrieveTopR(query, db, r)
      assertResultsEqual(results, expected, s"queryIndex=$idx")
    }
  }

  test("retrieveTopR edge case r=0 returns empty") {
    val query = SparseHash(Array(0), 1)
    val db = IndexedSeq(SparseHash(Array(1), 1))
    assertEquals(Retrieval.retrieveTopR(query, db, 0), IndexedSeq.empty)
  }

  test("retrieveTopR edge case empty database returns empty") {
    val query = SparseHash(Array(0), 1)
    assertEquals(Retrieval.retrieveTopR(query, IndexedSeq.empty, 5), IndexedSeq.empty)
  }

  test("retrieveTopR edge case r exceeds database length returns all") {
    val query = SparseHash(Array(0, 1), 2)
    val db = IndexedSeq(
      SparseHash(Array(2, 3), 2),
      SparseHash(Array(0, 2), 2)
    )
    val expected = RetrievalOracle.retrieveTopR(query, db, 10)
    val actual = Retrieval.retrieveTopR(query, db, 10)
    assertResultsEqual(actual, expected, "r > db.length")
  }

  test("retrieveTopR edge case all indices excluded returns empty") {
    val query = SparseHash(Array(0), 1)
    val db = IndexedSeq(SparseHash(Array(0), 1), SparseHash(Array(1), 1))
    assertEquals(Retrieval.retrieveTopR(query, db, 2, excludeIndices = Set(0, 1)), IndexedSeq.empty)
  }

  test("retrieveTopR edge case single-item database") {
    val rng = Random(123L)
    val query = HashGenerators.randomSparseHash(rng, 2, 8)
    val db = IndexedSeq(HashGenerators.randomSparseHash(rng, 2, 8))
    val expected = RetrievalOracle.retrieveTopR(query, db, 1)
    val actual = Retrieval.retrieveTopR(query, db, 1)
    assertResultsEqual(actual, expected, "single-item db")
  }

  test("retrieveTopR edge case tied distances") {
    val query = SparseHash(Array(0, 1), 2)
    val db = IndexedSeq(
      SparseHash(Array(2, 3), 2), // distance 4
      SparseHash(Array(4, 5), 2), // distance 4
      SparseHash(Array(0, 2), 2) // distance 2
    )
    val expected = RetrievalOracle.retrieveTopR(query, db, 3)
    val actual = Retrieval.retrieveTopR(query, db, 3)
    assertResultsEqual(actual, expected, "tied distances")
  }
