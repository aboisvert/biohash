// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.*
import io.github.aboisvert.biohash.data.*
import io.github.aboisvert.biohash.testkit.DenseRetrievalOracle
import java.nio.file.Files
import scala.compiletime.uninitialized

/** Layer C: text semantic diagnostics vs dense cosine oracle.
  *
  * Failure guide:
  *   - Layer A failure → retrieval/heap/encode bug
  *   - Layer C rank-1 failure on mini fixture → hash quality or training config
  *   - Layer C failure only on multi-segment → expected merge limitation; consolidate
  *   - Runner vs Service mismatch → TextBenchmarkRunner.query path bug
  */
class TextSemanticCorrectnessSuite extends munit.FunSuite:

  private val fixtureConfig =
    EvalConfig(k = 4, activity = 0.5, epochs = 5, seed = 7L, normalizeInputs = true)

  private val textSplitEvalConfig =
    EvalConfig(k = 4, activity = 0.1, epochs = 10, seed = 7L, normalizeInputs = true)

  private var fixtureDir: java.nio.file.Path = uninitialized
  private var dataset: TextBenchmarkDataset = uninitialized
  private var artifactDir: java.nio.file.Path = uninitialized
  private var normalizedCorpus: IndexedSeq[Array[Double]] = uninitialized

  override def beforeAll(): Unit =
    fixtureDir = TextBenchmarkFixtures.writeFixture(Files.createTempDirectory("biohash-text-semantic-fixture"))
    dataset = TextBenchmark.load(fixtureDir, Some("mini"))
    normalizedCorpus = dataset.corpusVectors.map(VectorOps.l2NormalizeInput)
    val artifactRoot = Files.createTempDirectory("biohash-text-semantic-artifacts")
    artifactDir = TextBenchmarkRunner.train(dataset, fixtureConfig, artifactRoot).artifactDir

  private def service: TextSearchService = TextSearchService.load(artifactDir)

  private def assertDocIdsEqual(
      queryId: String,
      queryVector: Array[Double],
      actual: IndexedSeq[String],
      expected: IndexedSeq[String]
  ): Unit =
    if actual != expected then
      fail(
        DenseRetrievalOracle.formatRankingDiagnostic(
          queryId = queryId,
          denseDocIds = expected,
          hammingDocIds = actual,
          corpusIds = dataset.corpusIds,
          query = queryVector,
          corpus = dataset.corpusVectors,
          normalizedCorpus = normalizedCorpus
        )
      )

  private def runnerDocIds(retrievalLimit: Int): IndexedSeq[IndexedSeq[String]] =
    val artifact = TextIndexArtifact.load(artifactDir)
    if artifact.segmentCount == 1 then
      val segment = artifact.latestSegment
      val queryHashes = segment.encoder.encodeAll(dataset.queryVectors)
      queryHashes.map { queryHash =>
        Retrieval
          .retrieveTopR(queryHash, segment.corpusHashes, retrievalLimit)
          .map(result => segment.corpusIds(result.index))
      }
    else
      dataset.queryVectors.map { queryVector =>
        SegmentedRetrieval.retrieveTopR(queryVector, artifact.segments, retrievalLimit)
      }

  test("dense oracle rank-1 is d1 for q1 and d3 for q2") {
    val q1 = dataset.queryVectors(0)
    val q2 = dataset.queryVectors(1)
    assertEquals(DenseRetrievalOracle.retrieveTopRDocIds(q1, dataset.corpusVectors, dataset.corpusIds, 1, normalizedCorpus).head, "d1")
    assertEquals(DenseRetrievalOracle.retrieveTopRDocIds(q2, dataset.corpusVectors, dataset.corpusIds, 1, normalizedCorpus).head, "d3")
  }

  test("BioHash rank-1 matches dense cosine on mini fixture") {
    dataset.queryIds.zip(dataset.queryVectors).foreach { case (queryId, queryVector) =>
      val expected = DenseRetrievalOracle.retrieveTopRDocIds(
        queryVector,
        dataset.corpusVectors,
        dataset.corpusIds,
        1,
        normalizedCorpus
      )
      val actual = service.search(queryVector, k = 1).map(_.docId)
      assertDocIdsEqual(queryId, queryVector, actual, expected)
    }
  }

  test("BioHash top-2 on q1 matches dense cosine order") {
    val queryId = dataset.queryIds.head
    val queryVector = dataset.queryVectors.head
    val expected = DenseRetrievalOracle.retrieveTopRDocIds(
      queryVector,
      dataset.corpusVectors,
      dataset.corpusIds,
      2,
      normalizedCorpus
    )
    val actual = service.search(queryVector, k = 2).map(_.docId)
    assertEquals(expected.toSeq, Seq("d1", "d2"))
    assertDocIdsEqual(queryId, queryVector, actual, expected)
  }

  test("TextBenchmarkRunner agrees with TextSearchService on mini fixture") {
    val limit = 3
    val fromRunner = runnerDocIds(limit)
    val fromService = dataset.queryVectors.map(vector => service.search(vector, k = limit).map(_.docId))
    assertEquals(fromRunner, fromService)
  }

  test("search returns monotonically non-decreasing Hamming distances") {
    dataset.queryVectors.foreach { queryVector =>
      val distances = service.search(queryVector, k = dataset.corpusSize).map(_.hamming)
      distances.sliding(2).foreach {
        case Seq(prev, next) => assert(prev <= next, s"Hamming not monotonic: $prev > $next")
        case _               => ()
      }
    }
  }

  test("corpus self-search returns matching doc with Hamming 0") {
    dataset.corpusIds.zip(dataset.corpusVectors).foreach { case (docId, vector) =>
      val hits = service.search(vector, k = dataset.corpusSize)
      val selfHit = hits.find(_.docId == docId)
      assert(selfHit.isDefined, s"self-search missing $docId in results: ${hits.map(_.docId).mkString(", ")}")
      assertEquals(selfHit.get.hamming, 0, s"self-search Hamming for $docId")
    }
  }

  test("axis-aligned corpus docs are rank-1 self-hits") {
    val axisAligned = Set("d1", "d3", "d4")
    dataset.corpusIds.zip(dataset.corpusVectors).collect { case (docId, vector) if axisAligned.contains(docId) =>
      (docId, vector)
    }.foreach { case (docId, vector) =>
      val hits = service.search(vector, k = 1)
      assertEquals(hits.head.docId, docId, s"rank-1 self-search for $docId")
      assertEquals(hits.head.hamming, 0, s"self-search Hamming for $docId")
    }
  }

  test("ranking correlation between dense cosine and Hamming on mini fixture") {
    dataset.queryIds.zip(dataset.queryVectors).foreach { case (queryId, queryVector) =>
      val denseOrder = DenseRetrievalOracle.retrieveTopRByCosine(
        queryVector,
        dataset.corpusVectors,
        dataset.corpusSize,
        normalizedCorpus
      )
      val hammingOrder = service.search(queryVector, k = dataset.corpusSize).map(_.docId)
      val correlation = spearmanCorrelation(denseOrder, hammingOrder, dataset.corpusIds)
      assert(
        correlation >= 0.75,
        s"query $queryId Spearman=$correlation\n${DenseRetrievalOracle.formatRankingDiagnostic(
            queryId,
            denseOrder.map(dataset.corpusIds(_)),
            hammingOrder,
            dataset.corpusIds,
            queryVector,
            dataset.corpusVectors,
            normalizedCorpus
          )}"
      )
    }
  }

  test("consolidated artifact rank-1 still matches dense cosine") {
    val artifactRoot = Files.createTempDirectory("biohash-text-semantic-consolidate")
    val localDataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val trained = TextBenchmarkRunner.train(localDataset, fixtureConfig, artifactRoot)
    val newIds = IndexedSeq("d5")
    val newVectors = IndexedSeq(Array(0.1, 0.9, 0.0))
    TextBenchmarkRunner.improveEncoder(trained.artifactDir, newIds, newVectors)
    val vectorsById =
      localDataset.corpusIds.zip(localDataset.corpusVectors).toMap + ("d5" -> newVectors.head)
    TextBenchmarkRunner.consolidate(trained.artifactDir, vectorsById)

    val consolidatedService = TextSearchService.load(trained.artifactDir)
    localDataset.queryIds.zip(localDataset.queryVectors).foreach { case (queryId, queryVector) =>
      val expected = DenseRetrievalOracle.retrieveTopRDocIds(
        queryVector,
        localDataset.corpusVectors,
        localDataset.corpusIds,
        1,
        localDataset.corpusVectors.map(VectorOps.l2NormalizeInput)
      )
      val actual = consolidatedService.search(queryVector, k = 1).map(_.docId)
      assertDocIdsEqual(queryId, queryVector, actual, expected)
    }
  }

  test("multi-segment merge reports when global rank-1 differs from dense") {
    val artifactRoot = Files.createTempDirectory("biohash-text-semantic-multiseg")
    val localDataset = TextBenchmark.load(fixtureDir, Some("mini"))
    val trained = TextBenchmarkRunner.train(localDataset, fixtureConfig, artifactRoot)
    val newIds = IndexedSeq("d5")
    val newVectors = IndexedSeq(Array(0.1, 0.9, 0.0))
    TextBenchmarkRunner.improveEncoder(trained.artifactDir, newIds, newVectors)

    val artifact = TextIndexArtifact.load(trained.artifactDir)
    assertEquals(artifact.segmentCount, 2)

    val normalized = localDataset.corpusVectors.map(VectorOps.l2NormalizeInput)
    localDataset.queryIds.zip(localDataset.queryVectors).foreach { case (queryId, queryVector) =>
      artifact.segments.foreach { segment =>
        val queryHash = segment.encoder.encode(queryVector)
        val oracle = Retrieval.retrieveTopR(queryHash, segment.corpusHashes, segment.corpusHashes.length)
        val candidates = SegmentedRetrieval.segmentCandidates(queryVector, segment, normalizeCrossSegmentDistances = false)
        assertEquals(candidates.map(_.corpusId), oracle.map(result => segment.corpusIds(result.index)))
      }

      val denseRank1 = DenseRetrievalOracle.retrieveTopRDocIds(
        queryVector,
        localDataset.corpusVectors,
        localDataset.corpusIds,
        1,
        normalized
      ).head
      val multiSegRank1 =
        SegmentedRetrieval.retrieveTopR(queryVector, artifact.segments, r = 1).head
      if multiSegRank1 != denseRank1 then
        fail(
          s"multi-segment merge may miss global nearest for $queryId: dense rank-1=$denseRank1, merged rank-1=$multiSegRank1"
        )
    }
  }

  private val textSplitConfig =
    TextRetrievalConfig(
      corpusSize = 40,
      queryCount = 10,
      dim = 16,
      topics = 4,
      clustersPerTopic = 2,
      noise = 0.15,
      hardNegativeRate = 0.0,
      seed = 99L
    )

  test("textRetrievalSplit dense rank-1 recall is high on paraphrase queries") {
    val split = Synthetic.textRetrievalSplit(textSplitConfig)
    val normalizedDb = split.databaseVectors.map(VectorOps.l2NormalizeInput)
    val groundTruth = split.queryVectors.map { query =>
      DenseRetrievalOracle.retrieveTopRByCosine(query, split.databaseVectors, 1, normalizedDb).head
    }
    val denseRetrieved = split.queryVectors.map { query =>
      DenseRetrievalOracle.retrieveTopRByCosine(query, split.databaseVectors, 1, normalizedDb)
    }
    val denseRecall = Metrics.recallAtR(denseRetrieved, groundTruth, r = 1)
    // fixture: seed=99, corpus=40, queries=10, dim=16, topics=4
    assertEqualsDouble(denseRecall, 1.0, 1e-9, "dense rank-1 recall")
  }

  test("textRetrievalSplit BioHash rank-1 recall meets pinned threshold") {
    val split = Synthetic.textRetrievalSplit(textSplitConfig)
    val normalizedDb = split.databaseVectors.map(VectorOps.l2NormalizeInput)
    val groundTruth = split.queryVectors.map { query =>
      DenseRetrievalOracle.retrieveTopRByCosine(query, split.databaseVectors, 1, normalizedDb).head
    }

    val m = math.ceil(textSplitEvalConfig.k / textSplitEvalConfig.activity).toInt
    val bh = new BioHash(
      BioHashConfig.paper(
        inputDim = textSplitConfig.dim,
        m = m,
        k = textSplitEvalConfig.k,
        learningRate = textSplitEvalConfig.learningRate,
        epochs = textSplitEvalConfig.epochs,
        seed = textSplitEvalConfig.seed,
        normalizeInputs = true
      )
    )
    bh.train(split.trainVectors)
    val dbHashes = bh.encodeAll(split.databaseVectors)
    val queryHashes = bh.encodeAll(split.queryVectors)
    val retrieved = queryHashes.map(q => Retrieval.retrieveTopR(q, dbHashes, 1).map(_.index))
    val recall = Metrics.recallAtR(retrieved, groundTruth, r = 1)
    // fixture: seed=99, corpus=40, queries=10, dim=16, k=4, activity=0.1, epochs=10, seed=7
    assertEqualsDouble(recall, 0.4, 1e-9, "BioHash rank-1 recall")
  }

  test("textRetrievalSplit BioHash rank-1 recall is at least FlyHash") {
    val split = Synthetic.textRetrievalSplit(textSplitConfig)
    val normalizedDb = split.databaseVectors.map(VectorOps.l2NormalizeInput)
    val groundTruth = split.queryVectors.map { query =>
      DenseRetrievalOracle.retrieveTopRByCosine(query, split.databaseVectors, 1, normalizedDb).head
    }

    def rank1Recall(method: HashMethod): Double =
      val encoder: HashEncoder = method match
        case HashMethod.BioHash =>
          val m = math.ceil(textSplitEvalConfig.k / textSplitEvalConfig.activity).toInt
          val bh = new BioHash(
            BioHashConfig.paper(
              inputDim = textSplitConfig.dim,
              m = m,
              k = textSplitEvalConfig.k,
              learningRate = textSplitEvalConfig.learningRate,
              epochs = textSplitEvalConfig.epochs,
              seed = textSplitEvalConfig.seed,
              normalizeInputs = true
            )
          )
          bh.train(split.trainVectors)
          bh
        case HashMethod.FlyHash =>
          new FlyHash(
            FlyHashConfig(inputDim = textSplitConfig.dim, m = 64, k = textSplitEvalConfig.k, seed = textSplitEvalConfig.seed)
          )
        case _ => throw IllegalArgumentException("unsupported method")
      val dbHashes = encoder.encodeAll(split.databaseVectors)
      val queryHashes = encoder.encodeAll(split.queryVectors)
      val retrieved = queryHashes.map(q => Retrieval.retrieveTopR(q, dbHashes, 1).map(_.index))
      Metrics.recallAtR(retrieved, groundTruth, r = 1)

    val bioHashRecall = rank1Recall(HashMethod.BioHash)
    val flyHashRecall = rank1Recall(HashMethod.FlyHash)
    assertEqualsDouble(bioHashRecall, 0.4, 1e-9, "BioHash rank-1 recall")
    assertEqualsDouble(flyHashRecall, 0.3, 1e-9, "FlyHash rank-1 recall")
    assert(
      bioHashRecall >= flyHashRecall,
      s"expected BioHash rank-1 recall ($bioHashRecall) >= FlyHash ($flyHashRecall)"
    )
  }

  private def spearmanCorrelation(
      denseOrder: IndexedSeq[Int],
      hammingDocIds: IndexedSeq[String],
      corpusIds: IndexedSeq[String]
  ): Double =
    val n = corpusIds.length
    if n <= 1 then 1.0
    else
      val denseRanks = denseOrder.zipWithIndex.toMap.view.mapValues(_ + 1).toMap
      val hammingRanks = hammingDocIds.zipWithIndex.map { case (docId, idx) =>
        corpusIds.indexOf(docId) -> (idx + 1)
      }.toMap
      val indices = corpusIds.indices
      val denseRankList = indices.map(denseRanks.getOrElse(_, n)).toVector
      val hammingRankList = indices.map(hammingRanks.getOrElse(_, n)).toVector
      val meanDense = denseRankList.sum.toDouble / n
      val meanHamming = hammingRankList.sum.toDouble / n
      var num = 0.0
      var denDense = 0.0
      var denHamming = 0.0
      var i = 0
      while i < n do
        val d = denseRankList(i) - meanDense
        val h = hammingRankList(i) - meanHamming
        num += d * h
        denDense += d * d
        denHamming += h * h
        i += 1
      if denDense == 0.0 || denHamming == 0.0 then 1.0
      else num / math.sqrt(denDense * denHamming)
