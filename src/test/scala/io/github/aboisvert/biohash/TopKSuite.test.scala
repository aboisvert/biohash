// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.testkit.{HashGenerators, RetrievalOracle}
import scala.util.Random

class TopKSuite extends munit.FunSuite:

  test("topKIndices selects largest scores") {
    val scores = Array(1.0, 5.0, 3.0, 2.0)
    assertEquals(TopK.topKIndices(scores, 2).toSeq, Seq(1, 2))
  }

  test("topKIndices breaks ties by lower index") {
    val scores = Array(5.0, 5.0, 1.0)
    assertEquals(TopK.topKIndices(scores, 2).toSeq, Seq(0, 1))
  }

  test("topKIndices k=0 returns empty") {
    assertEquals(TopK.topKIndices(Array(1.0, 2.0), 0).length, 0)
  }

  test("topKIndices k=m returns all sorted") {
    assertEquals(TopK.topKIndices(Array(3.0, 1.0, 2.0), 3).toSeq, Seq(0, 1, 2))
  }

  test("rankIndices descending") {
    val scores = Array(1.0, 5.0, 3.0)
    assertEquals(TopK.rankIndices(scores).toSeq, Seq(1, 2, 0))
  }

  test("topKIndices matches rankIndices take k") {
    val scores = Array(8.86, 6.54, -7.49, -1.75, -1.27, -2.96, 9.95, 6.73, 7.85, -8.26,
      4.65, -1.92, -3.64, -2.00, -8.40, 9.01, -4.96, -3.49, 1.74, -7.49,
      -6.68, -0.52, 0.72, -6.31)
    assertEquals(TopK.topKIndices(scores, 2).toSeq, TopK.rankIndices(scores).take(2).sorted.toSeq)
  }

  test("topKIndices matches oracle over random inputs") {
    val rng = Random(55L)
    var iteration = 0
    while iteration < 100 do
      val m = rng.nextInt(32) + 1
      val scores = HashGenerators.randomScores(rng, m)
      val k = rng.nextInt(m + 1)
      val expected = RetrievalOracle.topKIndices(scores, k)
      val actual = TopK.topKIndices(scores, k)
      assertEquals(actual.toSeq, expected.toSeq, s"iteration=$iteration m=$m k=$k")
      iteration += 1
  }
