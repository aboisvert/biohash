package biohash

object VectorOps:

  def dot(a: Array[Double], b: Array[Double]): Double =
    require(a.length == b.length, "dot: dimension mismatch")
    var sum = 0.0
    var i = 0
    while i < a.length do
      sum += a(i) * b(i)
      i += 1
    sum

  def pNorm(v: Array[Double], p: Double): Double =
    if p == 1.0 then
      var sum = 0.0
      var i = 0
      while i < v.length do
        sum += math.abs(v(i))
        i += 1
      sum
    else if p == 2.0 then
      var sum = 0.0
      var i = 0
      while i < v.length do
        sum += v(i) * v(i)
        i += 1
      math.sqrt(sum)
    else
      var sum = 0.0
      var i = 0
      while i < v.length do
        sum += math.pow(math.abs(v(i)), p)
        i += 1
      math.pow(sum, 1.0 / p)

  def normalizeInPlace(v: Array[Double], p: Double): Unit =
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
    rows.map(row => weightedScore(row, x, p))
