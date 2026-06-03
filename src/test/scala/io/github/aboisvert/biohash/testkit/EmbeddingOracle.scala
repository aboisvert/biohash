// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.testkit

/** Brute-force nearest-neighbor ground truth in embedding space (L2). */
object EmbeddingOracle:

  def l2Squared(a: Array[Double], b: Array[Double]): Double =
    require(a.length == b.length, "l2Squared: dimension mismatch")
    var sum = 0.0
    var i = 0
    while i < a.length do
      val d = a(i) - b(i)
      sum += d * d
      i += 1
    sum

  /** Index of the nearest database vector by L2 distance; ties broken by lower index. */
  def nearestL2(query: Array[Double], database: IndexedSeq[Array[Double]]): Int =
    require(database.nonEmpty, "nearestL2: database must be non-empty")
    database.indices.minBy(i => (l2Squared(query, database(i)), i))

  def groundTruthNearest(
      queries: IndexedSeq[Array[Double]],
      database: IndexedSeq[Array[Double]]
  ): IndexedSeq[Int] =
    queries.map(q => nearestL2(q, database))
