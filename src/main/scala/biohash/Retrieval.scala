package biohash

final case class RetrievalResult(index: Int, distance: Int)

object Retrieval:

  /** Exact top-R retrieval by Hamming distance; ties broken by lower index. */
  def retrieveTopR(
      query: SparseHash,
      database: IndexedSeq[SparseHash],
      r: Int,
      excludeIndices: Set[Int] = Set.empty
  ): IndexedSeq[RetrievalResult] =
    val limit = math.min(r, database.length)
    database.zipWithIndex
      .filterNot { (_, idx) => excludeIndices.contains(idx) }
      .map { (hash, idx) =>
        RetrievalResult(idx, SparseHash.hammingDistance(query, hash))
      }
      .sortBy(res => (res.distance, res.index))
      .take(limit)

  def retrieveAll(
      query: SparseHash,
      database: IndexedSeq[SparseHash],
      excludeIndices: Set[Int] = Set.empty
  ): IndexedSeq[RetrievalResult] =
    retrieveTopR(query, database, database.length, excludeIndices)

  def batchRetrieveTopR(
      queries: IndexedSeq[SparseHash],
      database: IndexedSeq[SparseHash],
      r: Int
  ): IndexedSeq[IndexedSeq[RetrievalResult]] =
    queries.map(q => retrieveTopR(q, database, r))
