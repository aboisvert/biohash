// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.bench

import io.github.aboisvert.biohash.*
import io.github.aboisvert.biohash.data.Synthetic
import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import scala.compiletime.uninitialized

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(1)
@State(Scope.Benchmark)
class BioHashJmh:

  private var bh: BioHash = uninitialized
  private var sample: Array[Double] = uninitialized
  private var scores: Array[Double] = uninitialized
  private var hash: SparseHash = uninitialized
  private var dbHashes: IndexedSeq[SparseHash] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val dim = 64
    val data = Synthetic.randomUnitVectors(200, dim, seed = 1L)
    bh = new BioHash(BioHashConfig.paper(inputDim = dim, m = 128, k = 4, epochs = 1, seed = 1L))
    bh.train(data)
    sample = data.head
    scores = bh.scores(sample)
    hash = bh.encode(sample)
    dbHashes = bh.encodeAll(data)

  @Benchmark
  def score(): Unit =
    bh.scores(sample)

  @Benchmark
  def topK(): Unit =
    TopK.topKIndices(scores, 4)

  @Benchmark
  def encode(): Unit =
    bh.encode(sample)

  @Benchmark
  def hamming(bhole: Blackhole): Unit =
    bhole.consume(SparseHash.hammingDistance(hash, dbHashes(1)))

  @Benchmark
  def retrieveTop100(): Unit =
    Retrieval.retrieveTopR(hash, dbHashes, 100)
