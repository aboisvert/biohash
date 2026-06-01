// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.Retrieval

/** One retrieval candidate from a specific index segment. */
final case class SegmentRetrievalCandidate(
    segmentId: Int,
    corpusId: String,
    distance: Int,
    normalizedDistance: Option[Double] = None
)

/** Cross-segment retrieval helpers. Hamming distances across segments are only approximately comparable. */
object SegmentedRetrieval:

  private given candidateOrdering: Ordering[SegmentRetrievalCandidate] =
    new Ordering[SegmentRetrievalCandidate]:
      def compare(a: SegmentRetrievalCandidate, b: SegmentRetrievalCandidate): Int =
        (a.normalizedDistance, b.normalizedDistance) match
          case (Some(na), Some(nb)) =>
            val byNormalized = na.compareTo(nb)
            if byNormalized != 0 then byNormalized else compareRaw(a, b)
          case _ => compareRaw(a, b)

      private def compareRaw(a: SegmentRetrievalCandidate, b: SegmentRetrievalCandidate): Int =
        val byDistance = a.distance.compare(b.distance)
        if byDistance != 0 then byDistance
        else
          val bySegment = a.segmentId.compare(b.segmentId)
          if bySegment != 0 then bySegment else a.corpusId.compare(b.corpusId)

  /** Retrieve top-R corpus ids across all segments for one query vector. */
  def retrieveTopR(
      queryVector: Array[Double],
      segments: IndexedSeq[TextIndexSegment],
      r: Int,
      normalizeCrossSegmentDistances: Boolean = false
  ): IndexedSeq[String] =
    val candidates =
      segments.flatMap(segmentCandidates(queryVector, _, normalizeCrossSegmentDistances))
    mergeCandidates(candidates, r)

  def segmentCandidates(
      queryVector: Array[Double],
      segment: TextIndexSegment,
      normalizeCrossSegmentDistances: Boolean
  ): IndexedSeq[SegmentRetrievalCandidate] =
    val queryHash = segment.encoder.encode(queryVector)
    val raw = Retrieval
      .retrieveTopR(queryHash, segment.corpusHashes, segment.corpusHashes.length)
      .map { result =>
        SegmentRetrievalCandidate(
          segmentId = segment.segmentId,
          corpusId = segment.corpusIds(result.index),
          distance = result.distance
        )
      }
    if !normalizeCrossSegmentDistances || raw.isEmpty then raw
    else
      val minDistance = raw.map(_.distance).min
      val maxDistance = raw.map(_.distance).max
      val span = (maxDistance - minDistance).toDouble
      raw.map { candidate =>
        val normalized =
          if span == 0.0 then 0.0
          else (candidate.distance - minDistance).toDouble / span
        candidate.copy(normalizedDistance = Some(normalized))
      }

  def mergeCandidates(
      candidates: Iterable[SegmentRetrievalCandidate],
      r: Int
  ): IndexedSeq[String] =
    candidates.toSeq
      .sorted
      .take(r)
      .map(_.corpusId)
      .toIndexedSeq
