// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.testkit

import io.github.aboisvert.biohash.VectorOps

/** Brute-force dense retrieval oracle matching [[io.github.aboisvert.biohash.eval.TextBenchmarkRunner]] baseline. */
object DenseRetrievalOracle:

  def cosineScores(
      query: Array[Double],
      corpus: IndexedSeq[Array[Double]],
      normalizedCorpus: IndexedSeq[Array[Double]]
  ): IndexedSeq[(Int, Double)] =
    val normalizedQuery = VectorOps.l2NormalizeInput(query)
    corpus.indices.map { idx =>
      (idx, VectorOps.dot(normalizedQuery, normalizedCorpus(idx)))
    }

  /** Corpus indices of top-R docs by cosine similarity; ties broken by lower index. */
  def retrieveTopRByCosine(
      query: Array[Double],
      corpus: IndexedSeq[Array[Double]],
      r: Int,
      normalizedCorpus: IndexedSeq[Array[Double]] = IndexedSeq.empty
  ): IndexedSeq[Int] =
    val docs =
      if normalizedCorpus.length == corpus.length then normalizedCorpus
      else corpus.map(VectorOps.l2NormalizeInput)
    val limit = math.min(r, corpus.length)
    if limit == 0 then IndexedSeq.empty
    else
      cosineScores(query, corpus, docs)
        .sortBy { case (idx, score) => (-score, idx) }
        .take(limit)
        .map(_._1)
        .toIndexedSeq

  def retrieveTopRDocIds(
      query: Array[Double],
      corpus: IndexedSeq[Array[Double]],
      corpusIds: IndexedSeq[String],
      r: Int,
      normalizedCorpus: IndexedSeq[Array[Double]] = IndexedSeq.empty
  ): IndexedSeq[String] =
    retrieveTopRByCosine(query, corpus, r, normalizedCorpus).map(corpusIds(_))

  def nearestCosine(
      query: Array[Double],
      corpus: IndexedSeq[Array[Double]],
      normalizedCorpus: IndexedSeq[Array[Double]] = IndexedSeq.empty
  ): Int =
    retrieveTopRByCosine(query, corpus, 1, normalizedCorpus).head

  def formatRankingDiagnostic(
      queryId: String,
      denseDocIds: IndexedSeq[String],
      hammingDocIds: IndexedSeq[String],
      corpusIds: IndexedSeq[String],
      query: Array[Double],
      corpus: IndexedSeq[Array[Double]],
      normalizedCorpus: IndexedSeq[Array[Double]] = IndexedSeq.empty
  ): String =
    val docs =
      if normalizedCorpus.length == corpus.length then normalizedCorpus
      else corpus.map(VectorOps.l2NormalizeInput)
    val scores = cosineScores(query, corpus, docs)
      .map { case (idx, score) => (corpusIds(idx), score) }
      .sortBy { case (id, score) => (-score, id) }
    val scoreLines = scores.map { case (id, score) => f"  $id: cosine=$score%.4f" }.mkString("\n")
    s"""query $queryId ranking mismatch
       |dense top:  ${denseDocIds.mkString("[", ", ", "]")}
       |hamming top: ${hammingDocIds.mkString("[", ", ", "]")}
       |cosine scores (all corpus docs):
       |$scoreLines""".stripMargin
