// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

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
