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
      val heap = new Array[Int](k)
      var size = 0
      var idx = 0
      while idx < scores.length do
        if size < k then
          heap(size) = idx
          siftUp(heap, scores, size)
          size += 1
        else if isBetter(scores, idx, heap(0)) then
          heap(0) = idx
          siftDown(heap, scores, size, 0)
        idx += 1
      java.util.Arrays.sort(heap, 0, size)
      java.util.Arrays.copyOf(heap, size)

  /** Indices sorted by score descending; ties broken by lower index first. */
  def rankIndices(scores: Array[Double]): Array[Int] =
    val indices = Array.tabulate(scores.length)(identity)
    indices.sortInPlace()(using Ordering.by((i: Int) => (-scores(i), i)))
    indices

  /** Indices of the top `ranks` scores in rank order (best first). `ranks` is 1-based. */
  def topRankedIndices(scores: Array[Double], ranks: Int): Array[Int] =
    require(ranks >= 1 && ranks <= scores.length, s"topRankedIndices: ranks=$ranks out of range")
    val heap = new Array[Int](ranks)
    var size = 0
    var idx = 0
    while idx < scores.length do
      if size < ranks then
        heap(size) = idx
        siftUp(heap, scores, size)
        size += 1
      else if isBetter(scores, idx, heap(0)) then
        heap(0) = idx
        siftDown(heap, scores, size, 0)
      idx += 1

    val selected = java.util.Arrays.copyOf(heap, size)
    selected.sortInPlace()(using Ordering.by((i: Int) => (-scores(i), i)))
    selected

  /** True when index `a` should rank above index `b`. */
  private def isBetter(scores: Array[Double], a: Int, b: Int): Boolean =
    val scoreCmp = java.lang.Double.compare(scores(a), scores(b))
    scoreCmp > 0 || (scoreCmp == 0 && a < b)

  /** Min-heap on the worst current candidate (lowest score; tie -> higher index). */
  private def worseThan(scores: Array[Double], a: Int, b: Int): Boolean =
    val scoreCmp = java.lang.Double.compare(scores(a), scores(b))
    scoreCmp < 0 || (scoreCmp == 0 && a > b)

  private def siftUp(heap: Array[Int], scores: Array[Double], pos: Int): Unit =
    var child = pos
    while child > 0 do
      val parent = (child - 1) >>> 1
      if worseThan(scores, heap(child), heap(parent)) then
        val tmp = heap(parent)
        heap(parent) = heap(child)
        heap(child) = tmp
        child = parent
      else child = -1

  private def siftDown(heap: Array[Int], scores: Array[Double], size: Int, pos: Int): Unit =
    var parent = pos
    var continue = true
    while continue do
      val left = parent * 2 + 1
      if left >= size then continue = false
      else
        var worst = left
        val right = left + 1
        if right < size && worseThan(scores, heap(right), heap(worst)) then worst = right
        if worseThan(scores, heap(parent), heap(worst)) then
          val tmp = heap(parent)
          heap(parent) = heap(worst)
          heap(worst) = tmp
          parent = worst
        else continue = false
