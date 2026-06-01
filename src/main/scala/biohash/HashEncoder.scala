// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash

/** Common interface for hash encoders used in evaluation and retrieval. */
trait HashEncoder:
  def encode(x: Array[Double]): SparseHash
  def encodeAll(data: IndexedSeq[Array[Double]]): IndexedSeq[SparseHash] =
    data.map(encode)
