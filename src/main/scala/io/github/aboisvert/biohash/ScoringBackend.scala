// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

/** Numeric kernel backend used by [[VectorOps]] and [[WeightMatrix]] scoring.
  */
trait ScoringBackend:
  def name: String
  def dot(a: Array[Double], b: Array[Double]): Double
  def pNormL2(v: Array[Double]): Double
  def normalizeInPlaceL2(v: Array[Double]): Unit
  def scoresGemv(
      matrix: WeightMatrix,
      x: Array[Double],
      p: Double,
      out: Array[Double]
  ): Unit

object ScoringBackend:

  private var overrideBackend: Option[ScoringBackend] = None

  def current: ScoringBackend =
    overrideBackend.getOrElse(resolveFromProperty)

  lazy val default: ScoringBackend =
    if VectorApiBackend.isSupported then VectorApiBackend
    else ScalarBackend

  def withBackend[A](backend: ScoringBackend)(f: => A): A =
    val previous = overrideBackend
    overrideBackend = Some(backend)
    try f
    finally overrideBackend = previous

  private def resolveFromProperty: ScoringBackend =
    // use BIOHASH_BACKEND environment variable if set, or default to the system property
    sys.env.get("BIOHASH_BACKEND").orElse(sys.props.get("biohash.backend")) match
      case Some("scalar") => ScalarBackend
      case Some("vector") => VectorApiBackend
      case Some("blas")   => BlasBackend
      case Some(other)    => throw IllegalArgumentException(s"Unknown biohash.backend: $other")
      case None           => default

/** Reference scalar loops. */
object ScalarBackend extends ScoringBackend:
  val name = "scalar"

  def dot(a: Array[Double], b: Array[Double]): Double =
    require(a.length == b.length, "dot: dimension mismatch")
    var sum = 0.0
    var i = 0
    while i < a.length do
      sum += a(i) * b(i)
      i += 1
    sum

  def pNormL2(v: Array[Double]): Double =
    math.sqrt(dot(v, v))

  def normalizeInPlaceL2(v: Array[Double]): Unit =
    val norm = pNormL2(v)
    if norm > 0.0 then
      var i = 0
      while i < v.length do
        v(i) /= norm
        i += 1

  def scoresGemv(
      matrix: WeightMatrix,
      x: Array[Double],
      p: Double,
      out: Array[Double]
  ): Unit =
    require(out.length == matrix.rows, "scoresGemv: output length must equal matrix rows")
    if p == 2.0 then
      var row = 0
      while row < matrix.rows do
        val offset = matrix.rowOffset(row)
        var sum = 0.0
        var col = 0
        while col < matrix.cols do
          sum += matrix.flatData(offset + col) * x(col)
          col += 1
        out(row) = sum
        row += 1
    else
      var row = 0
      while row < matrix.rows do
        val offset = matrix.rowOffset(row)
        var sum = 0.0
        var col = 0
        while col < matrix.cols do
          val w = matrix.flatData(offset + col)
          sum += math.pow(math.abs(w), p - 2.0) * w * x(col)
          col += 1
        out(row) = sum
        row += 1

/** JDK Vector API (incubator) backend. */
object VectorApiBackend extends ScoringBackend:
  val name = "vector"

  def isSupported: Boolean =
    try
      Class.forName("jdk.incubator.vector.DoubleVector")
      Class.forName("io.github.aboisvert.biohash.VectorApiBackendImpl")
      true
    catch case _: Throwable => false

  private lazy val impl: ScoringBackend =
    Class
      .forName("io.github.aboisvert.biohash.VectorApiBackendImpl$")
      .getField("MODULE$")
      .get(null)
      .asInstanceOf[ScoringBackend]

  def dot(a: Array[Double], b: Array[Double]): Double = impl.dot(a, b)
  def pNormL2(v: Array[Double]): Double = impl.pNormL2(v)
  def normalizeInPlaceL2(v: Array[Double]): Unit = impl.normalizeInPlaceL2(v)
  def scoresGemv(
      matrix: WeightMatrix,
      x: Array[Double],
      p: Double,
      out: Array[Double]
  ): Unit = impl.scoresGemv(matrix, x, p, out)

/** BLAS backend via netlib-java (pure Java fallback, native when present). */
object BlasBackend extends ScoringBackend:
  val name = "blas"

  private val blas = dev.ludovic.netlib.blas.BLAS.getInstance()

  def dot(a: Array[Double], b: Array[Double]): Double =
    require(a.length == b.length, "dot: dimension mismatch")
    blas.ddot(a.length, a, 1, b, 1)

  def pNormL2(v: Array[Double]): Double =
    math.sqrt(blas.ddot(v.length, v, 1, v, 1))

  def normalizeInPlaceL2(v: Array[Double]): Unit =
    val norm = pNormL2(v)
    if norm > 0.0 then blas.dscal(v.length, 1.0 / norm, v, 1)

  def scoresGemv(
      matrix: WeightMatrix,
      x: Array[Double],
      p: Double,
      out: Array[Double]
  ): Unit =
    require(out.length == matrix.rows, "scoresGemv: output length must equal matrix rows")
    if p == 2.0 then
      var row = 0
      while row < matrix.rows do
        val offset = matrix.rowOffset(row)
        out(row) = blas.ddot(matrix.cols, matrix.flatData, offset, 1, x, 0, 1)
        row += 1
    else ScalarBackend.scoresGemv(matrix, x, p, out)
