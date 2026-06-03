// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.testkit

import io.github.aboisvert.biohash.SparseHash
import scala.util.Random

object HashGenerators:

  def randomSparseHash(rng: Random, k: Int, m: Int): SparseHash =
    require(k > 0 && k <= m, s"randomSparseHash: require 0 < k <= m, got k=$k m=$m")
    val indices = rng.shuffle((0 until m).toList).take(k).sorted.toArray
    SparseHash(indices, k)

  def randomDatabase(rng: Random, n: Int, k: Int, m: Int): IndexedSeq[SparseHash] =
    IndexedSeq.tabulate(n)(_ => randomSparseHash(rng, k, m))

  def randomScores(rng: Random, m: Int): Array[Double] =
    Array.tabulate(m)(_ => rng.nextDouble() * 20.0 - 10.0)

  def randomExclude(rng: Random, databaseSize: Int): Set[Int] =
    val count = rng.nextInt(databaseSize + 1)
    rng.shuffle((0 until databaseSize).toList).take(count).toSet
