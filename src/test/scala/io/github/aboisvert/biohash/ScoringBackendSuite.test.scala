// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

class ScoringBackendSuite extends munit.FunSuite:

  private val backends = Seq(ScalarBackend, VectorApiBackend, BlasBackend)

  test("backends agree on dot product") {
    val a = Array(0.1, -0.2, 0.3, 0.4)
    val b = Array(0.5, 0.6, -0.7, 0.8)
    val expected = ScalarBackend.dot(a, b)
    backends.foreach { backend =>
      assertEqualsDouble(backend.dot(a, b), expected, 1e-9, backend.name)
    }
  }

  test("backends agree on gemv scoring") {
    val matrix = WeightMatrix.fromNested(
      Array(
        Array(1.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0),
        Array(1.0, 1.0, 1.0)
      )
    )
    val input = Array(2.0, 3.0, 4.0)
    val expected = new Array[Double](3)
    ScalarBackend.scoresGemv(matrix, input, 2.0, expected)
    backends.foreach { backend =>
      val out = new Array[Double](3)
      backend.scoresGemv(matrix, input, 2.0, out)
      assertEquals(out.toSeq, expected.toSeq, backend.name)
    }
  }

  test("topRankedIndices returns ranks in order") {
    val scores = Array(1.0, 5.0, 3.0, 2.0)
    assertEquals(TopK.topRankedIndices(scores, 2).toSeq, Seq(1, 2))
  }

  test("topRankedIndices breaks ties by lower index") {
    val scores = Array(5.0, 5.0, 1.0)
    assertEquals(TopK.topRankedIndices(scores, 2).toSeq, Seq(0, 1))
  }

  test("WeightMatrix round-trips nested weights") {
    val nested = Array(Array(1.0, 2.0), Array(3.0, 4.0))
    val matrix = WeightMatrix.fromNested(nested)
    assertEquals(matrix.weights.toSeq.map(_.toSeq), nested.toSeq.map(_.toSeq))
  }
