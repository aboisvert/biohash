// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash

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
