// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

class BioHashSuite extends munit.FunSuite:

  test("encode returns exactly k active indices") {
    val config = BioHashConfig.paper(inputDim = 4, m = 8, k = 2, seed = 1L)
    val bh = new BioHash(config)
    val x = Array(1.0, 0.0, 0.0, 0.0)
    val hash = bh.encode(x)
    assertEquals(hash.k, 2)
    assertEquals(hash.active.length, 2)
  }

  test("weights remain normalized after training step") {
    val config = BioHashConfig.paper(inputDim = 4, m = 4, k = 2, learningRate = 0.1, seed = 1L)
    val bh = new BioHash(config)
    val x = Array(1.0, 0.5, 0.0, 0.0)
    bh.trainStep(x)
    bh.weights.foreach { row =>
      assertEqualsDouble(VectorOps.pNorm(row, 2.0), 1.0, 1e-6)
    }
  }

  test("winner moves toward input for delta=0") {
    val config = BioHashConfig.paper(inputDim = 2, m = 2, k = 1, learningRate = 0.5, delta = 0.0, seed = 0L)
    val bh = BioHash.fromWeights(
      config,
      Array(
        Array(1.0, 0.0),
        Array(0.0, 1.0)
      )
    )
    val x = Array(1.0, 0.0)
    val before = VectorOps.dot(bh.weights(0), x)
    bh.trainStep(x)
    val after = VectorOps.dot(bh.weights(0), x)
    assert(after >= before - 1e-9)
  }

  test("deterministic encoding with fixed seed") {
    val config = BioHashConfig.paper(inputDim = 8, m = 16, k = 2, seed = 99L)
    val data = Array.tabulate(10)(i => Array.tabulate(8)(j => (i + j).toDouble / 10.0))
    val bh1 = new BioHash(config)
    bh1.train(data.toIndexedSeq)
    val bh2 = new BioHash(config)
    bh2.train(data.toIndexedSeq)
    val h1 = bh1.encode(data(0))
    val h2 = bh2.encode(data(0))
    assertEquals(h1.active.toSeq, h2.active.toSeq)
  }

  test("trainMiniBatch changes weights and preserves row normalization") {
    val config = BioHashConfig.paper(inputDim = 4, m = 4, k = 2, learningRate = 0.1, seed = 5L)
    val bh = BioHash.fromWeights(
      config,
      Array(
        Array(1.0, 0.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0, 0.0),
        Array(0.0, 0.0, 1.0, 0.0),
        Array(0.0, 0.0, 0.0, 1.0)
      )
    )
    val before = bh.weights.map(_.clone())
    val batch = IndexedSeq(Array(1.0, 0.5, 0.0, 0.0), Array(0.0, 1.0, 0.5, 0.0))
    bh.trainMiniBatch(batch, epochs = 2, shuffle = false)
    assert(!bh.weights.zip(before).forall { case (after, prior) => after.sameElements(prior) })
    bh.weights.foreach { row =>
      assertEqualsDouble(VectorOps.pNorm(row, 2.0), 1.0, 1e-6)
    }
    assertEquals(bh.currentTrainingSteps, batch.length * 2L)
  }

  test("trainMiniBatch with shuffle=false matches repeated trainStep") {
    val config = BioHashConfig.paper(inputDim = 3, m = 4, k = 2, learningRate = 0.05, seed = 11L)
    val batch = IndexedSeq(Array(1.0, 0.0, 0.0), Array(0.0, 1.0, 0.0))
    val viaMiniBatch = BioHash.fromWeights(
      config,
      Array(
        Array(1.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0),
        Array(0.0, 0.0, 1.0),
        Array(1.0, 1.0, 0.0)
      )
    )
    viaMiniBatch.trainMiniBatch(batch, epochs = 2, shuffle = false)
    val viaSteps = BioHash.fromWeights(
      config,
      Array(
        Array(1.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0),
        Array(0.0, 0.0, 1.0),
        Array(1.0, 1.0, 0.0)
      )
    )
    batch.foreach(v => viaSteps.trainStep(v))
    batch.foreach(v => viaSteps.trainStep(v))
    assert(viaMiniBatch.weights.zip(viaSteps.weights).forall { case (a, b) => a.sameElements(b) })
  }
