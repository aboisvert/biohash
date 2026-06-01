// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash

import biohash.data.*
import biohash.eval.*

class FlyHashSuite extends munit.FunSuite:

  test("FlyHash encode returns k active indices") {
    val fh = new FlyHash(FlyHashConfig(inputDim = 10, m = 50, k = 3, seed = 1L))
    val x = Array.fill(10)(0.5)
    val hash = fh.encode(x)
    assertEquals(hash.k, 3)
  }

  test("NaiveBioHash trains and encodes") {
    val data = Synthetic.twoClassBlobs(dim = 8, perClass = 20, seed = 1L)
    val split = DatasetSplit.byClass(data, queriesPerClass = 5, seed = 1L)
    val nb = new NaiveBioHash(NaiveBioHashConfig.paper(inputDim = 8, k = 4, epochs = 2, seed = 1L))
    nb.train(split.trainVectors)
    val hash = nb.encode(split.queryVectors.head)
    assert(hash.active.length <= 4)
  }

class IntegrationSuite extends munit.FunSuite:

  test("BioHash improves same-class overlap on synthetic blobs") {
    val data = Synthetic.twoClassBlobs(dim = 16, perClass = 40, seed = 42L)
    val split = DatasetSplit.byClass(data, queriesPerClass = 10, seed = 42L)
    val config = EvalConfig(k = 2, activity = 0.1, epochs = 10, seed = 42L, method = HashMethod.BioHash)
    val result = EvalRunner.runSplit(split, config, "synthetic")
    assert(result.mAP > 0.3, s"expected mAP > 0.3, got ${result.mAP}")
  }

  test("FlyHash baseline runs on synthetic data") {
    val data = Synthetic.twoClassBlobs(dim = 8, perClass = 20, seed = 1L)
    val split = DatasetSplit.byClass(data, queriesPerClass = 5, seed = 1L)
    val config = EvalConfig(k = 2, activity = 0.1, epochs = 1, seed = 1L, method = HashMethod.FlyHash)
    val result = EvalRunner.runSplit(split, config, "synthetic")
    assert(result.mAP >= 0.0)
  }

  test("BenchmarkRunner microbenchmarks complete") {
    val results = BenchmarkRunner.runMicrobenchmarks(dim = 16, m = 32, k = 2, n = 50)
    assertEquals(results.length, 5)
    results.foreach(r => assert(r.opsPerSecond > 0))
  }

class DatasetSplitSuite extends munit.FunSuite:

  test("byClass produces correct query count") {
    val data = Synthetic.twoClassBlobs(dim = 4, perClass = 30, seed = 0L)
    val split = DatasetSplit.byClass(data, queriesPerClass = 10, seed = 0L)
    assertEquals(split.queryVectors.length, 20) // 2 classes * 10
    assertEquals(split.databaseVectors.length, 40) // 60 - 20
  }
