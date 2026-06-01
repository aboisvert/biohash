// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

object VectorOps:

  private def backend: ScoringBackend = ScoringBackend.current

  def dot(a: Array[Double], b: Array[Double]): Double =
    backend.dot(a, b)

  def pNorm(v: Array[Double], p: Double): Double =
    if p == 1.0 then
      var sum = 0.0
      var i = 0
      while i < v.length do
        sum += math.abs(v(i))
        i += 1
      sum
    else if p == 2.0 then
      backend.pNormL2(v)
    else
      var sum = 0.0
      var i = 0
      while i < v.length do
        sum += math.pow(math.abs(v(i)), p)
        i += 1
      math.pow(sum, 1.0 / p)

  def normalizeInPlace(v: Array[Double], p: Double): Unit =
    if p == 2.0 then backend.normalizeInPlaceL2(v)
    else
      val norm = pNorm(v, p)
      if norm > 0.0 then
        var i = 0
        while i < v.length do
          v(i) /= norm
          i += 1

  def normalizedCopy(v: Array[Double], p: Double): Array[Double] =
    val out = v.clone()
    normalizeInPlace(out, p)
    out

  def l2NormalizeInput(v: Array[Double]): Array[Double] =
    normalizedCopy(v, 2.0)

  /** Metric-weighted score for row mu when p != 2: sum_i |W_i|^{p-2} * W_i * x_i */
  def weightedScore(row: Array[Double], x: Array[Double], p: Double): Double =
    if p == 2.0 then dot(row, x)
    else
      var sum = 0.0
      var i = 0
      while i < row.length do
        val w = row(i)
        sum += math.pow(math.abs(w), p - 2.0) * w * x(i)
        i += 1
      sum

  def scoresMatrix(rows: Array[Array[Double]], x: Array[Double], p: Double): Array[Double] =
    val matrix = WeightMatrix.fromNested(rows)
    val out = new Array[Double](matrix.rows)
    backend.scoresGemv(matrix, x, p, out)
    out

  def scoresMatrix(matrix: WeightMatrix, x: Array[Double], p: Double, out: Array[Double]): Unit =
    backend.scoresGemv(matrix, x, p, out)
