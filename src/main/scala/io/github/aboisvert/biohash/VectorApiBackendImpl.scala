// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import jdk.incubator.vector.*

/** JDK Vector API implementation loaded only when the incubator module is available. */
object VectorApiBackendImpl extends ScoringBackend:
  val name = "vector"

  private val species = DoubleVector.SPECIES_PREFERRED

  def dot(a: Array[Double], b: Array[Double]): Double =
    require(a.length == b.length, "dot: dimension mismatch")
    var sumVec = DoubleVector.zero(species)
    var i = 0
    val upper = species.loopBound(a.length)
    while i < upper do
      sumVec = sumVec.add(
        DoubleVector.fromArray(species, a, i).mul(DoubleVector.fromArray(species, b, i))
      )
      i += species.length
    var sum = sumVec.reduceLanes(VectorOperators.ADD)
    while i < a.length do
      sum += a(i) * b(i)
      i += 1
    sum

  def pNormL2(v: Array[Double]): Double =
    math.sqrt(dot(v, v))

  def normalizeInPlaceL2(v: Array[Double]): Unit =
    val norm = pNormL2(v)
    if norm > 0.0 then
      val inv = 1.0 / norm
      val invVec = DoubleVector.broadcast(species, inv)
      var i = 0
      val upper = species.loopBound(v.length)
      while i < upper do
        DoubleVector
          .fromArray(species, v, i)
          .mul(invVec)
          .intoArray(v, i)
        i += species.length
      while i < v.length do
        v(i) *= inv
        i += 1

  def scoresGemv(
      matrix: WeightMatrix,
      x: Array[Double],
      p: Double,
      out: Array[Double]
  ): Unit =
    if p == 2.0 then
      var row = 0
      while row < matrix.rows do
        val offset = matrix.rowOffset(row)
        var sumVec = DoubleVector.zero(species)
        var col = 0
        val upper = species.loopBound(matrix.cols)
        while col < upper do
          sumVec = sumVec.add(
            DoubleVector
              .fromArray(species, matrix.flatData, offset + col)
              .mul(DoubleVector.fromArray(species, x, col))
          )
          col += species.length
        var sum = sumVec.reduceLanes(VectorOperators.ADD)
        while col < matrix.cols do
          sum += matrix.flatData(offset + col) * x(col)
          col += 1
        out(row) = sum
        row += 1
    else ScalarBackend.scoresGemv(matrix, x, p, out)
