// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.testkit.RetrievalOracle
import scala.util.Random

class EncodeCorrectnessSuite extends munit.FunSuite:

  private val backends = Seq(ScalarBackend, VectorApiBackend, BlasBackend)

  private def sampleBioHash: BioHash =
    BioHash.fromWeights(
      BioHashConfig.paper(inputDim = 4, m = 8, k = 3, seed = 1L, normalizeInputs = false),
      Array(
        Array(1.0, 0.0, 0.0, 0.0),
        Array(0.0, 1.0, 0.0, 0.0),
        Array(0.0, 0.0, 1.0, 0.0),
        Array(0.0, 0.0, 0.0, 1.0),
        Array(1.0, 1.0, 0.0, 0.0),
        Array(0.0, 1.0, 1.0, 0.0),
        Array(0.0, 0.0, 1.0, 1.0),
        Array(1.0, 0.0, 1.0, 0.0)
      ),
      trainingSteps = 1L
    )

  private def expectedTopK(encoder: BioHash, x: Array[Double]): Array[Int] =
    val scores = encoder.scores(x)
    RetrievalOracle.topKIndices(scores, encoder.config.k)

  test("BioHash encode matches scalar top-k oracle") {
    val encoder = sampleBioHash
    val rng = Random(11L)
    var iteration = 0
    while iteration < 50 do
      val x = Array.tabulate(4)(_ => rng.nextGaussian())
      val expected = expectedTopK(encoder, x)
      val actual = encoder.encode(x).active
      assertEquals(actual.toSeq, expected.toSeq, s"iteration=$iteration")
      iteration += 1
  }

  test("BioHash encode agrees across scoring backends") {
    val encoder = sampleBioHash
    val rng = Random(22L)
    val x = Array.tabulate(4)(_ => rng.nextGaussian())
    val expected = ScoringBackend.withBackend(ScalarBackend)(encoder.encode(x).active.toSeq)
    backends.foreach { backend =>
      val actual = ScoringBackend.withBackend(backend)(encoder.encode(x).active.toSeq)
      assertEquals(actual, expected, backend.name)
    }
  }

  test("FlyHash encode matches top-k oracle") {
    val encoder = new FlyHash(FlyHashConfig(inputDim = 6, m = 12, k = 3, samplingRate = 0.5, seed = 3L))
    val rng = Random(33L)
    var iteration = 0
    while iteration < 30 do
      val x = Array.tabulate(6)(_ => rng.nextGaussian())
      val scores = encoder.scores(x)
      val expected = RetrievalOracle.topKIndices(scores, encoder.config.k)
      val actual = encoder.encode(x).active
      assertEquals(actual.toSeq, expected.toSeq, s"iteration=$iteration")
      iteration += 1
  }

  test("NaiveBioHash encode matches sign oracle") {
    val encoder = new NaiveBioHash(NaiveBioHashConfig.paper(inputDim = 4, k = 4, seed = 4L, epochs = 1))
    encoder.train(IndexedSeq(Array(1.0, 0.0, 0.0, 0.0), Array(0.0, 1.0, 0.0, 0.0)))
    val rng = Random(44L)
    var iteration = 0
    while iteration < 20 do
      val x = Array.tabulate(4)(_ => rng.nextGaussian())
      val scores = encoder.innerBioHash.scores(x)
      val expected = scores.zipWithIndex.collect { case (score, idx) if score > 0 => idx }.sorted
      val actual = encoder.encode(x).active
      assertEquals(actual.toSeq, expected.toSeq, s"iteration=$iteration")
      iteration += 1
  }
