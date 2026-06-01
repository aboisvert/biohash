package biohash

class VectorOpsSuite extends munit.FunSuite:

  test("dot product") {
    val a = Array(1.0, 2.0, 3.0)
    val b = Array(4.0, 5.0, 6.0)
    assertEqualsDouble(VectorOps.dot(a, b), 32.0, 1e-9)
  }

  test("l2 norm of unit vector") {
    val v = Array(3.0, 4.0)
    assertEqualsDouble(VectorOps.pNorm(v, 2.0), 5.0, 1e-9)
  }

  test("l1 norm") {
    val v = Array(-1.0, 2.0, -3.0)
    assertEqualsDouble(VectorOps.pNorm(v, 1.0), 6.0, 1e-9)
  }

  test("normalizeInPlace yields unit l2 norm") {
    val v = Array(3.0, 4.0)
    VectorOps.normalizeInPlace(v, 2.0)
    assertEqualsDouble(VectorOps.pNorm(v, 2.0), 1.0, 1e-9)
  }

  test("l2NormalizeInput returns new array") {
    val v = Array(3.0, 4.0)
    val out = VectorOps.l2NormalizeInput(v)
    assert(out ne v)
    assertEqualsDouble(VectorOps.pNorm(out, 2.0), 1.0, 1e-9)
  }

  test("weighted score equals dot for p=2") {
    val row = Array(1.0, 0.0)
    val x = Array(0.6, 0.8)
    assertEqualsDouble(VectorOps.weightedScore(row, x, 2.0), 0.6, 1e-9)
  }
