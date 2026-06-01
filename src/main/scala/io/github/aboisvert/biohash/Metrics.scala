// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

object Metrics:

  /** Average precision at cutoff R for one query. labels: relevance 0/1 per database item. */
  def averagePrecision(
      retrievedIndices: IndexedSeq[Int],
      labels: IndexedSeq[Int],
      r: Int
  ): Double =
    require(labels.forall(l => l == 0 || l == 1), "labels must be 0/1")
    val cutoff = math.min(r, retrievedIndices.length)
    val totalRelevant = labels.sum
    if totalRelevant == 0 then 0.0
    else
      var hits = 0
      var sumPrecision = 0.0
      var l = 0
      while l < cutoff do
        if labels(retrievedIndices(l)) == 1 then
          hits += 1
          sumPrecision += hits.toDouble / (l + 1)
        l += 1
      sumPrecision / totalRelevant

  def mAP(
      queryResults: IndexedSeq[IndexedSeq[Int]],
      relevanceLabels: IndexedSeq[IndexedSeq[Int]],
      r: Int
  ): Double =
    require(queryResults.length == relevanceLabels.length)
    if queryResults.isEmpty then 0.0
    else
      val aps = queryResults.zip(relevanceLabels).map { (retrieved, labels) =>
        averagePrecision(retrieved, labels, r)
      }
      aps.sum / aps.length

  /** Recall@R: fraction of queries whose 1-NN (by ground truth) appears in top R. */
  def recallAtR(
      retrievedIndices: IndexedSeq[IndexedSeq[Int]],
      groundTruthNearest: IndexedSeq[Int],
      r: Int
  ): Double =
    require(retrievedIndices.length == groundTruthNearest.length)
    if retrievedIndices.isEmpty then 0.0
    else
      val hits = retrievedIndices.zip(groundTruthNearest).count { (retrieved, gt) =>
        retrieved.take(r).contains(gt)
      }
      hits.toDouble / retrievedIndices.length

  def evaluateRetrieval(
      queries: IndexedSeq[SparseHash],
      database: IndexedSeq[SparseHash],
      queryLabels: IndexedSeq[Int],
      databaseLabels: IndexedSeq[Int],
      r: Int
  ): Double =
    val results = queries.map { q =>
      Retrieval.retrieveTopR(q, database, r).map(_.index)
    }
    val relevance = queryLabels.map { ql =>
      databaseLabels.map(l => if l == ql then 1 else 0)
    }
    mAP(results, relevance, r)
