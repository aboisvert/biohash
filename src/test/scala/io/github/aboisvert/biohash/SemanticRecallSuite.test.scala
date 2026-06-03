// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.data.*
import io.github.aboisvert.biohash.eval.{EvalConfig, HashMethod}
import io.github.aboisvert.biohash.testkit.EmbeddingOracle
import java.nio.file.Files

/** Layer B tests: hash quality vs brute-force L2 1-NN ground truth (not retrieval algorithm bugs). */
class SemanticRecallSuite extends munit.FunSuite:

  private val fixtureSeed = 42L
  private val evalConfig =
    EvalConfig(k = 2, activity = 0.1, epochs = 10, seed = fixtureSeed, method = HashMethod.BioHash)

  private def toySplit: RetrievalSplit =
    val data = Synthetic.twoClassBlobs(dim = 8, perClass = 20, seed = fixtureSeed)
    DatasetSplit.byClass(data, queriesPerClass = 5, seed = fixtureSeed)

  private def recallAt(
      method: HashMethod,
      split: RetrievalSplit,
      config: EvalConfig,
      r: Int
  ): Double =
    val inputDim = split.trainVectors.head.length
    val m = math.ceil(config.k / config.activity).toInt
    val encoder: HashEncoder = method match
      case HashMethod.BioHash =>
        val bh = new BioHash(
          BioHashConfig.paper(
            inputDim = inputDim,
            m = m,
            k = config.k,
            learningRate = config.learningRate,
            epochs = config.epochs,
            seed = config.seed,
            normalizeInputs = config.normalizeInputs
          )
        )
        bh.train(split.trainVectors)
        bh
      case HashMethod.FlyHash =>
        new FlyHash(FlyHashConfig.paperBaseline(inputDim, config.k, config.seed))
      case HashMethod.NaiveBioHash =>
        val nb = new NaiveBioHash(
          NaiveBioHashConfig.paper(
            inputDim = inputDim,
            k = config.k,
            learningRate = config.learningRate,
            epochs = config.epochs,
            seed = config.seed,
            normalizeInputs = config.normalizeInputs
          )
        )
        nb.train(split.trainVectors)
        nb

    val dbHashes = encoder.encodeAll(split.databaseVectors)
    val queryHashes = encoder.encodeAll(split.queryVectors)
    val retrieved = queryHashes.map(q => Retrieval.retrieveTopR(q, dbHashes, r).map(_.index))
    val groundTruth = EmbeddingOracle.groundTruthNearest(split.queryVectors, split.databaseVectors)
    Metrics.recallAtR(retrieved, groundTruth, r)

  test("BioHash recall@10 meets threshold on toy ANN fixture") {
    val split = toySplit
    val recall = recallAt(HashMethod.BioHash, split, evalConfig, r = 10)
    // fixture: seed=42, dim=8, perClass=20, queriesPerClass=5, k=2, activity=0.1, epochs=10
    // Lower bound rather than exact pin: regression guard catches drops below 0.8,
    // while improvements (recall > 0.8) correctly pass.
    // Note: recall@10 on a ~30-item DB is a weak test; random floor = 10/30 ≈ 0.33.
    // SemanticQualityAuditSuite exercises stronger recall@1 and multi-seed controls.
    assert(recall >= 0.8, s"recall@10=$recall must be >= 0.8 on toy ANN fixture")
  }

  test("mini TEXMEX-style vectors recall@5 meets pinned threshold") {
    val dir = Files.createTempDirectory("biohash-semantic-ann")
    try
      val database = IndexedSeq(
        Array(1.0, 0.0),
        Array(0.0, 1.0),
        Array(1.0, 1.0),
        Array(-1.0, 0.0),
        Array(0.0, -1.0)
      )
      val queries = IndexedSeq(
        Array(0.9, 0.1),
        Array(0.1, 0.9),
        Array(0.8, 0.8)
      )
      val groundTruth = EmbeddingOracle.groundTruthNearest(queries, database)
      assertEquals(groundTruth.toSeq, Seq(0, 1, 2))

      val bh = new BioHash(
        BioHashConfig.paper(inputDim = 2, m = 20, k = 2, epochs = 5, seed = 7L, normalizeInputs = true)
      )
      bh.train(database)
      val dbHashes = bh.encodeAll(database)
      val queryHashes = bh.encodeAll(queries)
      val retrieved = queryHashes.map(q => Retrieval.retrieveTopR(q, dbHashes, 5).map(_.index))
      val recall = Metrics.recallAtR(retrieved, groundTruth, r = 5)
      // fixture: 5 db vectors, 3 queries, dim=2, k=2, m=20, seed=7, epochs=5
      assertEqualsDouble(recall, 1.0, 1e-9, "recall@5")
    finally Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
  }

  test("FlyHash recall is at most BioHash on toy ANN fixture (negative control)") {
    val split = toySplit
    val bioHashRecall = recallAt(HashMethod.BioHash, split, evalConfig, r = 10)
    val flyHashRecall = recallAt(HashMethod.FlyHash, split, evalConfig, r = 10)
    assert(
      flyHashRecall <= bioHashRecall,
      s"expected FlyHash recall ($flyHashRecall) <= BioHash recall ($bioHashRecall)"
    )
    assert(bioHashRecall > 0.0, "BioHash should retrieve at least one true nearest neighbor")
  }
