package biohash.eval

import biohash.*
import biohash.data.Synthetic

/** End-to-end and component runtime benchmarks (non-JMH; suitable for CI smoke tests). */
object BenchmarkRunner:

  final case class BenchResult(name: String, iterations: Int, totalNanos: Long):
    def opsPerSecond: Double = iterations.toDouble / (totalNanos / 1e9)

  def time[T](name: String, iterations: Int)(f: => T): BenchResult =
    // warmup
    var w = 0
    while w < math.min(100, iterations / 10) do { f; w += 1 }
    val start = System.nanoTime()
    var i = 0
    while i < iterations do
      f
      i += 1
    BenchResult(name, iterations, System.nanoTime() - start)

  def runMicrobenchmarks(dim: Int = 64, m: Int = 128, k: Int = 4, n: Int = 1000): IndexedSeq[BenchResult] =
    val data = Synthetic.randomUnitVectors(n, dim, seed = 1L)
    val config = BioHashConfig.paper(inputDim = dim, m = m, k = k, epochs = 1, seed = 1L)
    val bh = new BioHash(config)
    bh.train(data.take(n / 2))

    val sample = data.head
    val scores = bh.scores(sample)
    val hash = bh.encode(sample)
    val dbHashes = bh.encodeAll(data)

    IndexedSeq(
      time("score", 10000)(bh.scores(sample)),
      time("topK", 10000)(TopK.topKIndices(scores, k)),
      time("encode", 5000)(bh.encode(sample)),
      time("hamming", 10000)(SparseHash.hammingDistance(hash, dbHashes(1))),
      time("retrieve100", 500)(Retrieval.retrieveTopR(hash, dbHashes, 100))
    )

  def format(results: IndexedSeq[BenchResult]): String =
    results.map(r => f"${r.name}%-14s ${r.opsPerSecond}%,.0f ops/s").mkString("\n")
