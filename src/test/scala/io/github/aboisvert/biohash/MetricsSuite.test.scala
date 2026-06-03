// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

class MetricsSuite extends munit.FunSuite:

  test("averagePrecision hand calculation") {
    // Retrieved order: relevant at positions 1 and 3 (1-indexed) -> indices 0 and 2
    val retrieved = IndexedSeq(0, 1, 2)
    val labels = IndexedSeq(1, 0, 1)
    // AP = (1/1 + 2/3) / 2 = (1 + 0.666...) / 2
    val ap = Metrics.averagePrecision(retrieved, labels, r = 3)
    assertEqualsDouble(ap, (1.0 + 2.0 / 3.0) / 2.0, 1e-9)
  }

  test("averagePrecision zero when no relevant items") {
    val ap = Metrics.averagePrecision(IndexedSeq(0, 1), IndexedSeq(0, 0), r = 2)
    assertEqualsDouble(ap, 0.0, 1e-9)
  }

  test("mAP averages over queries") {
    val results = IndexedSeq(
      IndexedSeq(0, 1),
      IndexedSeq(1, 0)
    )
    val labels = IndexedSeq(
      IndexedSeq(1, 0),
      IndexedSeq(0, 1)
    )
    val map = Metrics.mAP(results, labels, r = 2)
    assertEqualsDouble(map, 1.0, 1e-9)
  }

  test("recallAtR") {
    val retrieved = IndexedSeq(
      IndexedSeq(5, 1, 2),
      IndexedSeq(0, 1, 2)
    )
    val gt = IndexedSeq(1, 0)
    assertEqualsDouble(Metrics.recallAtR(retrieved, gt, r = 3), 1.0, 1e-9)
    assertEqualsDouble(Metrics.recallAtR(retrieved, gt, r = 1), 0.5, 1e-9)
  }

  test("recallAtTopK perfect retrieval") {
    // Each query retrieves exactly its true top-3
    val retrieved = IndexedSeq(IndexedSeq(0, 1, 2), IndexedSeq(3, 4, 5))
    val truth     = IndexedSeq(IndexedSeq(0, 1, 2), IndexedSeq(3, 4, 5))
    assertEqualsDouble(Metrics.recallAtTopK(retrieved, truth), 1.0, 1e-9)
  }

  test("recallAtTopK partial overlap") {
    // Query 0: retrieved {0,1,2}, truth {0,1,3} → 2/3 = 0.666...
    // Query 1: retrieved {4,5,6}, truth {4,7,8} → 1/3 = 0.333...
    // Mean = 0.5
    val retrieved = IndexedSeq(IndexedSeq(0, 1, 2), IndexedSeq(4, 5, 6))
    val truth     = IndexedSeq(IndexedSeq(0, 1, 3), IndexedSeq(4, 7, 8))
    assertEqualsDouble(Metrics.recallAtTopK(retrieved, truth), 0.5, 1e-9)
  }

  test("recallAtTopK no overlap is zero") {
    val retrieved = IndexedSeq(IndexedSeq(0, 1), IndexedSeq(2, 3))
    val truth     = IndexedSeq(IndexedSeq(4, 5), IndexedSeq(6, 7))
    assertEqualsDouble(Metrics.recallAtTopK(retrieved, truth), 0.0, 1e-9)
  }

  test("recallAtTopK empty is zero") {
    assertEqualsDouble(
      Metrics.recallAtTopK(IndexedSeq.empty, IndexedSeq.empty),
      0.0,
      1e-9
    )
  }

  test("meanReciprocalRank perfect rank-1") {
    // 1-NN is always at position 0 → each RR = 1.0
    val retrieved = IndexedSeq(IndexedSeq(3, 1, 2), IndexedSeq(5, 0, 4))
    val gt        = IndexedSeq(3, 5)
    assertEqualsDouble(Metrics.meanReciprocalRank(retrieved, gt), 1.0, 1e-9)
  }

  test("meanReciprocalRank mixed ranks") {
    // Query 0: gt=1 at position 1 (rank 2) → RR = 0.5
    // Query 1: gt=4 at position 2 (rank 3) → RR = 1/3
    val retrieved = IndexedSeq(IndexedSeq(0, 1, 2), IndexedSeq(3, 5, 4))
    val gt        = IndexedSeq(1, 4)
    val expected  = (0.5 + 1.0 / 3.0) / 2.0
    assertEqualsDouble(Metrics.meanReciprocalRank(retrieved, gt), expected, 1e-9)
  }

  test("meanReciprocalRank not found yields zero") {
    val retrieved = IndexedSeq(IndexedSeq(0, 1, 2))
    val gt        = IndexedSeq(9)
    assertEqualsDouble(Metrics.meanReciprocalRank(retrieved, gt), 0.0, 1e-9)
  }
