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

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@State(Scope.Benchmark)
class BioHashTextJmh:

  private var bh: BioHash = uninitialized
  private var sample: Array[Double] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val dim = 384
    val m = 3200
    val data = Synthetic.randomUnitVectors(512, dim, seed = 2L)
    bh = new BioHash(BioHashConfig.paper(inputDim = dim, m = m, k = 32, epochs = 1, seed = 2L))
    bh.train(data)
    sample = data.head

  @Benchmark
  def encode(): Unit =
    bh.encode(sample)

  @Benchmark
  def score(): Unit =
    bh.scores(sample)

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@State(Scope.Benchmark)
class FlyHashMnistJmh:

  private var fh: FlyHash = uninitialized
  private var sample: Array[Double] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val dim = 784
    val data = Synthetic.randomUnitVectors(128, dim, seed = 3L)
    fh = new FlyHash(FlyHashConfig.paperBaseline(inputDim = dim, k = 2, seed = 3L))
    sample = data.head

  @Benchmark
  def encode(): Unit =
    fh.encode(sample)

  @Benchmark
  def score(): Unit =
    fh.scores(sample)

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@State(Scope.Benchmark)
class RetrievalLargeJmh:

  private var query: SparseHash = uninitialized
  private var database: IndexedSeq[SparseHash] = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val dim = 128
    val data = Synthetic.randomUnitVectors(10_000, dim, seed = 4L)
    val bh = new BioHash(BioHashConfig.paper(inputDim = dim, m = 256, k = 8, epochs = 1, seed = 4L))
    bh.train(data.take(1000))
    database = bh.encodeAll(data)
    query = database.head

  @Benchmark
  def retrieveTop100(): Unit =
    Retrieval.retrieveTopR(query, database, 100)

  @Benchmark
  def batchRetrieveTop100(): Unit =
    Retrieval.batchRetrieveTopR(database.take(32), database, 100)

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(1)
@State(Scope.Benchmark)
class ScoringBackendJmh:

  private var matrix: WeightMatrix = uninitialized
  private var input: Array[Double] = uninitialized
  private var output: Array[Double] = uninitialized

  @Param(Array("scalar", "vector", "blas"))
  var backendName: String = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    val dim = 384
    val rows = 3200
    val nested = Array.tabulate(rows) { row =>
      Array.tabulate(dim) { col => ((row + col) % 17).toDouble / 17.0 }
    }
    matrix = WeightMatrix.fromNested(nested)
    input = Array.tabulate(dim)(i => i.toDouble / dim)
    output = new Array[Double](rows)
    ScoringBackend.withBackend(backendForName(backendName))(() => ())

  @Benchmark
  def scoresGemv(): Unit =
    val backend = backendForName(backendName)
    ScoringBackend.withBackend(backend) {
      backend.scoresGemv(matrix, input, 2.0, output)
    }

  private def backendForName(name: String): ScoringBackend =
    name match
      case "scalar" => ScalarBackend
      case "vector" => VectorApiBackend
      case "blas"   => BlasBackend
      case other    => throw IllegalArgumentException(s"Unknown backend: $other")
