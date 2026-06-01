// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

/** Row-major dense weight matrix backed by a single contiguous buffer. */
final class WeightMatrix(val rows: Int, val cols: Int, private val data: Array[Double]):

  require(rows >= 0 && cols >= 0)
  require(data.length == rows * cols, "WeightMatrix: data length must equal rows * cols")

  def this(rows: Int, cols: Int) =
    this(rows, cols, new Array[Double](rows * cols))

  def flatData: Array[Double] = data

  def rowOffset(row: Int): Int =
    require(row >= 0 && row < rows, s"WeightMatrix: row=$row out of range [0, $rows)")
    row * cols

  def get(row: Int, col: Int): Double =
    data(rowOffset(row) + col)

  def set(row: Int, col: Int, value: Double): Unit =
    data(rowOffset(row) + col) = value

  /** Nested row view for persistence and compatibility with existing callers. */
  def weights: Array[Array[Double]] =
    Array.tabulate(rows) { row =>
      val offset = rowOffset(row)
      java.util.Arrays.copyOfRange(data, offset, offset + cols)
    }

  def copyFromNested(nested: Array[Array[Double]]): Unit =
    require(nested.length == rows, "WeightMatrix.copyFromNested: row count mismatch")
    require(nested.forall(_.length == cols), "WeightMatrix.copyFromNested: column count mismatch")
    var row = 0
    while row < rows do
      System.arraycopy(nested(row), 0, data, rowOffset(row), cols)
      row += 1

object WeightMatrix:

  def fromNested(nested: Array[Array[Double]]): WeightMatrix =
    val rows = nested.length
    val cols = if rows == 0 then 0 else nested.head.length
    val matrix = new WeightMatrix(rows, cols)
    matrix.copyFromNested(nested)
    matrix

  def tabulate(rows: Int, cols: Int)(f: (Int, Int) => Double): WeightMatrix =
    val matrix = new WeightMatrix(rows, cols)
    var row = 0
    while row < rows do
      val offset = matrix.rowOffset(row)
      var col = 0
      while col < cols do
        matrix.data(offset + col) = f(row, col)
        col += 1
      row += 1
    matrix
