package biohash

class RetrievalSuite extends munit.FunSuite:

  test("retrieveTopR returns lowest Hamming distance first") {
    val query = SparseHash(Array(0, 1), 2)
    val db = IndexedSeq(
      SparseHash(Array(2, 3), 2), // distance 4
      SparseHash(Array(0, 2), 2), // distance 2
      SparseHash(Array(0, 1), 2)  // distance 0
    )
    val results = Retrieval.retrieveTopR(query, db, 2)
    assertEquals(results.map(_.index).toSeq, Seq(2, 1))
    assertEquals(results.map(_.distance).toSeq, Seq(0, 2))
  }

  test("retrieveTopR tie-breaks by lower index") {
    val query = SparseHash(Array(0, 1), 2)
    val db = IndexedSeq(
      SparseHash(Array(2, 3), 2),
      SparseHash(Array(4, 5), 2)
    )
    val results = Retrieval.retrieveTopR(query, db, 2)
    assertEquals(results.map(_.index).toSeq, Seq(0, 1))
  }

  test("excludeIndices skips query items in database") {
    val query = SparseHash(Array(0), 1)
    val db = IndexedSeq(SparseHash(Array(0), 1), SparseHash(Array(1), 1))
    val results = Retrieval.retrieveTopR(query, db, 1, excludeIndices = Set(0))
    assertEquals(results.head.index, 1)
  }
