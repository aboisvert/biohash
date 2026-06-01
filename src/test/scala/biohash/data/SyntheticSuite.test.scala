// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash.data

import biohash.VectorOps

class SyntheticSuite extends munit.FunSuite:

  private def l2Norm(v: Array[Double]): Double = VectorOps.pNorm(v, 2.0)

  test("textRetrievalSplit is deterministic for the same seed") {
    val config = TextRetrievalConfig(
      corpusSize = 200,
      queryCount = 20,
      dim = 16,
      topics = 10,
      clustersPerTopic = 2,
      seed = 99L
    )
    val a = Synthetic.textRetrievalSplit(config)
    val b = Synthetic.textRetrievalSplit(config)
    assertEquals(a.databaseVectors.length, b.databaseVectors.length)
    assertEquals(a.queryVectors.length, b.queryVectors.length)
    a.databaseVectors.zip(b.databaseVectors).foreach { (x, y) =>
      assertEquals(x.toSeq, y.toSeq)
    }
    a.queryVectors.zip(b.queryVectors).foreach { (x, y) =>
      assertEquals(x.toSeq, y.toSeq)
    }
    assertEquals(a.databaseLabels, b.databaseLabels)
    assertEquals(a.queryLabels, b.queryLabels)
  }

  test("textRetrievalSplit produces requested split sizes") {
    val config = TextRetrievalConfig(
      corpusSize = 500,
      queryCount = 50,
      dim = 32,
      topics = 25,
      clustersPerTopic = 3,
      seed = 1L
    )
    val split = Synthetic.textRetrievalSplit(config)
    assertEquals(split.databaseVectors.length, 500)
    assertEquals(split.trainVectors.length, 500)
    assertEquals(split.queryVectors.length, 50)
    assertEquals(split.databaseLabels.length, 500)
    assertEquals(split.queryLabels.length, 50)
    assertEquals(split.databaseVectors.head.length, 32)
    assertEquals(split.queryVectors.head.length, 32)
  }

  test("textRetrievalSplit vectors are L2-normalized") {
    val split = Synthetic.textRetrievalSplit(
      TextRetrievalConfig(corpusSize = 100, queryCount = 10, dim = 64, topics = 5, seed = 7L)
    )
    (split.databaseVectors ++ split.queryVectors).foreach { v =>
      assert(math.abs(l2Norm(v) - 1.0) < 1e-9, s"expected unit norm, got ${l2Norm(v)}")
    }
  }

  test("each query label has at least one relevant database item") {
    val split = Synthetic.textRetrievalSplit(
      TextRetrievalConfig(corpusSize = 300, queryCount = 30, dim = 16, topics = 15, seed = 3L)
    )
    val labelsInDb = split.databaseLabels.toSet
    split.queryLabels.foreach { ql =>
      assert(labelsInDb.contains(ql), s"query label $ql has no matching database label")
      assert(split.databaseLabels.contains(ql), s"no database passage with label $ql")
    }
  }
