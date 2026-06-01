// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

class TextRetrievalMetricsSuite extends munit.FunSuite:

  test("ndcgAt prefers higher-scored relevant documents earlier") {
    val relevant = Map("d1" -> 2, "d2" -> 1)
    val good = TextRetrievalMetrics.ndcgAt(IndexedSeq("d1", "d2", "d3"), relevant, k = 2)
    val bad = TextRetrievalMetrics.ndcgAt(IndexedSeq("d2", "d3", "d1"), relevant, k = 2)
    assert(good > bad)
  }

  test("averagePrecisionAt hand calculation") {
    val relevant = Map("a" -> 1, "c" -> 1)
    val retrieved = IndexedSeq("a", "b", "c")
    val ap = TextRetrievalMetrics.averagePrecisionAt(retrieved, relevant, k = 3)
    assertEqualsDouble(ap, (1.0 + 2.0 / 3.0) / 2.0, 1e-9)
  }

  test("macroRecallAt counts found relevant docs") {
    val evaluated = IndexedSeq(
      (Map("d1" -> 1, "d2" -> 1), IndexedSeq("d1", "x")),
      (Map("d3" -> 1), IndexedSeq("d3", "y"))
    )
    assertEqualsDouble(TextRetrievalMetrics.macroRecallAt(evaluated, k = 1), 2.0 / 3.0, 1e-9)
    assertEqualsDouble(TextRetrievalMetrics.macroRecallAt(evaluated, k = 2), 2.0 / 3.0, 1e-9)
  }

  test("evaluate aggregates query metrics") {
    val qrels = Map(
      "q1" -> Map("d1" -> 1),
      "q2" -> Map("d2" -> 1)
    )
    val retrieved = IndexedSeq(
      IndexedSeq("d1", "x"),
      IndexedSeq("d2", "y")
    )
    val metrics = TextRetrievalMetrics.evaluate(retrieved, IndexedSeq("q1", "q2"), qrels)
    assertEqualsDouble(metrics.recallAt10, 1.0, 1e-9)
    assert(metrics.mapAt100 > 0.99)
  }
