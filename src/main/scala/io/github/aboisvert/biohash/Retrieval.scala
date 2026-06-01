// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

final case class RetrievalResult(index: Int, distance: Int)

object Retrieval:

  private val parallelQueryThreshold = 32

  /** Exact top-R retrieval by Hamming distance; ties broken by lower index. */
  def retrieveTopR(
      query: SparseHash,
      database: IndexedSeq[SparseHash],
      r: Int,
      excludeIndices: Set[Int] = Set.empty
  ): IndexedSeq[RetrievalResult] =
    val limit = math.min(r, database.length)
    if limit == 0 then IndexedSeq.empty
    else
      val heapIndices = new Array[Int](limit)
      val heapDistances = new Array[Int](limit)
      var size = 0
      var idx = 0
      while idx < database.length do
        if !excludeIndices.contains(idx) then
          val distance = SparseHash.hammingDistance(query, database(idx))
          if size < limit then
            heapIndices(size) = idx
            heapDistances(size) = distance
            siftUp(heapIndices, heapDistances, size)
            size += 1
          else if isBetter(distance, idx, heapDistances(0), heapIndices(0)) then
            heapIndices(0) = idx
            heapDistances(0) = distance
            siftDown(heapIndices, heapDistances, size, 0)
        idx += 1

      val ordered = orderResults(heapIndices, heapDistances, size)
      ordered.map { case (index, distance) => RetrievalResult(index, distance) }.toIndexedSeq

  def retrieveAll(
      query: SparseHash,
      database: IndexedSeq[SparseHash],
      excludeIndices: Set[Int] = Set.empty
  ): IndexedSeq[RetrievalResult] =
    retrieveTopR(query, database, database.length, excludeIndices)

  def batchRetrieveTopR(
      queries: IndexedSeq[SparseHash],
      database: IndexedSeq[SparseHash],
      r: Int
  ): IndexedSeq[IndexedSeq[RetrievalResult]] =
    ParallelOps.parMap(queries, parallelQueryThreshold)(q => retrieveTopR(q, database, r))

  /** True when (distance, index) is lexicographically smaller. */
  private def isBetter(
      distance: Int,
      index: Int,
      worstDistance: Int,
      worstIndex: Int
  ): Boolean =
    distance < worstDistance || (distance == worstDistance && index < worstIndex)

  /** Max-heap on the worst current candidate (largest distance; tie -> larger index). */
  private def worseThan(
      distanceA: Int,
      indexA: Int,
      distanceB: Int,
      indexB: Int
  ): Boolean =
    distanceA > distanceB || (distanceA == distanceB && indexA > indexB)

  private def siftUp(
      heapIndices: Array[Int],
      heapDistances: Array[Int],
      pos: Int
  ): Unit =
    var child = pos
    while child > 0 do
      val parent = (child - 1) >>> 1
      if worseThan(
          heapDistances(child),
          heapIndices(child),
          heapDistances(parent),
          heapIndices(parent)
        )
      then
        swap(heapIndices, heapDistances, child, parent)
        child = parent
      else child = -1

  private def siftDown(
      heapIndices: Array[Int],
      heapDistances: Array[Int],
      size: Int,
      pos: Int
  ): Unit =
    var parent = pos
    var continue = true
    while continue do
      val left = parent * 2 + 1
      if left >= size then continue = false
      else
        var worst = left
        val right = left + 1
        if right < size && worseThan(
            heapDistances(right),
            heapIndices(right),
            heapDistances(worst),
            heapIndices(worst)
          )
        then worst = right
        if worseThan(
            heapDistances(parent),
            heapIndices(parent),
            heapDistances(worst),
            heapIndices(worst)
          )
        then
          swap(heapIndices, heapDistances, parent, worst)
          parent = worst
        else continue = false

  private def swap(
      heapIndices: Array[Int],
      heapDistances: Array[Int],
      a: Int,
      b: Int
  ): Unit =
    val tmpIndex = heapIndices(a)
    heapIndices(a) = heapIndices(b)
    heapIndices(b) = tmpIndex
    val tmpDistance = heapDistances(a)
    heapDistances(a) = heapDistances(b)
    heapDistances(b) = tmpDistance

  private def orderResults(
      heapIndices: Array[Int],
      heapDistances: Array[Int],
      size: Int
  ): Array[(Int, Int)] =
    val results = Array.tabulate(size) { i =>
      (heapIndices(i), heapDistances(i))
    }
    results.sortInPlace()(using Ordering.by((entry: (Int, Int)) => (entry._2, entry._1)))
    results
