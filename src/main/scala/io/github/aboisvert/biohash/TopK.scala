// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

object TopK:

  /** Indices of the k largest scores; ties broken by lower index first. Result sorted ascending. */
  def topKIndices(scores: Array[Double], k: Int): Array[Int] =
    require(k >= 0 && k <= scores.length, s"topKIndices: k=$k out of range [0, ${scores.length}]")
    if k == 0 then Array.empty
    else if k == scores.length then Array.tabulate(scores.length)(identity)
    else
      val indexed = scores.zipWithIndex
      val top = indexed.sortWith { (a, b) =>
        val scoreCmp = a._1.compare(b._1)
        if scoreCmp != 0 then scoreCmp > 0 else a._2 < b._2
      }.take(k).map(_._2).sorted
      top

  /** Indices sorted by score descending; ties broken by lower index first. */
  def rankIndices(scores: Array[Double]): Array[Int] =
    scores.zipWithIndex
      .sortWith { (a, b) =>
        val scoreCmp = a._1.compare(b._1)
        if scoreCmp != 0 then scoreCmp > 0 else a._2 < b._2
      }
      .map(_._2)
