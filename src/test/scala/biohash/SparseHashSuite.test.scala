package biohash

class SparseHashSuite extends munit.FunSuite:

  test("hamming distance identical codes is 0") {
    val h = SparseHash(Array(1, 5, 10), 3)
    assertEquals(SparseHash.hammingDistance(h, h), 0)
  }

  test("hamming distance disjoint k-sparse codes is 2k") {
    val a = SparseHash(Array(0, 1), 2)
    val b = SparseHash(Array(2, 3), 2)
    assertEquals(SparseHash.hammingDistance(a, b), 4)
  }

  test("intersection size for overlapping codes") {
    val a = Array(1, 3, 5)
    val b = Array(3, 4, 5)
    assertEquals(SparseHash.intersectionSize(a, b), 2)
  }

  test("hamming via partial overlap") {
    val a = SparseHash(Array(1, 3, 5), 3)
    val b = SparseHash(Array(3, 5, 7), 3)
    assertEquals(SparseHash.hammingDistance(a, b), 2) // 2k - 2*2 = 2
  }
