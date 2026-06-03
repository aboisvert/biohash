// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.testkit

import io.github.aboisvert.biohash.{RetrievalResult, SparseHash}

/** Naive reference implementations for retrieval correctness tests.
  *
  * Contract (must match [[io.github.aboisvert.biohash.Retrieval]]):
  *   - Distance: [[SparseHash.hammingDistance]]
  *   - Ranking: lexicographic `(distance asc, index asc)`
  *   - Output size: up to `min(r, database.length)` non-excluded items
  */
object RetrievalOracle:

  def retrieveTopR(
      query: SparseHash,
      database: IndexedSeq[SparseHash],
      r: Int,
      excludeIndices: Set[Int] = Set.empty
  ): IndexedSeq[RetrievalResult] =
    val limit = math.min(r, database.length)
    if limit == 0 then IndexedSeq.empty
    else
      database.indices
        .filterNot(excludeIndices.contains)
        .map { i =>
          RetrievalResult(i, SparseHash.hammingDistance(query, database(i)))
        }
        .sortBy(result => (result.distance, result.index))
        .take(limit)
        .toIndexedSeq

  /** Indices of the k largest scores; ties broken by lower index; result sorted ascending. */
  def topKIndices(scores: Array[Double], k: Int): Array[Int] =
    require(k >= 0 && k <= scores.length, s"topKIndices: k=$k out of range [0, ${scores.length}]")
    if k == 0 then Array.empty
    else if k == scores.length then Array.tabulate(scores.length)(identity)
    else
      scores.indices.toArray
        .sortBy(i => (-scores(i), i))
        .take(k)
        .sorted
