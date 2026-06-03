// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.Retrieval
import java.nio.file.Path

final case class SearchHit(docId: String, hamming: Int, rank: Int)

final class TextSearchService private (artifact: TextIndexArtifact):

  def manifest: TextBenchmarkManifest = artifact.manifest

  def search(queryVector: Array[Double], k: Int): IndexedSeq[SearchHit] =
    require(k > 0, "search: k must be positive")
    require(
      queryVector.length == artifact.manifest.inputDim,
      s"search: expected dim ${artifact.manifest.inputDim}, got ${queryVector.length}"
    )
    if artifact.segmentCount == 1 then searchSingleSegment(queryVector, k)
    else searchMultiSegment(queryVector, k)

  private def searchSingleSegment(queryVector: Array[Double], k: Int): IndexedSeq[SearchHit] =
    val segment = artifact.latestSegment
    val queryHash = segment.encoder.encode(queryVector)
    Retrieval
      .retrieveTopR(queryHash, segment.corpusHashes, k)
      .zipWithIndex
      .map { case (result, idx) =>
        SearchHit(docId = segment.corpusIds(result.index), hamming = result.distance, rank = idx + 1)
      }

  private def searchMultiSegment(queryVector: Array[Double], k: Int): IndexedSeq[SearchHit] =
    import SegmentedRetrieval.candidateOrdering
    val candidates =
      artifact.segments.flatMap(SegmentedRetrieval.segmentCandidates(queryVector, _, normalizeCrossSegmentDistances = false))
    candidates.toSeq
      .sorted
      .take(k)
      .zipWithIndex
      .map { case (candidate, idx) =>
        SearchHit(docId = candidate.corpusId, hamming = candidate.distance, rank = idx + 1)
      }

object TextSearchService:

  def load(artifactDir: Path): TextSearchService =
    new TextSearchService(TextIndexArtifact.load(artifactDir))
