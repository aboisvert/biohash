package biohash

/** Sparse binary hash storing sorted indices of active (+1) bits. */
final case class SparseHash(active: Array[Int], k: Int):

  require(active.length == k, s"SparseHash: expected $k active indices, got ${active.length}")
  require(active.sameElements(active.sorted.distinct), "SparseHash: active indices must be sorted and unique")

object SparseHash:

  def apply(activeIndices: Array[Int]): SparseHash =
    SparseHash(activeIndices.clone(), activeIndices.length)

  /** Hamming distance between two k-sparse codes via sorted intersection. O(k). */
  def hammingDistance(a: SparseHash, b: SparseHash): Int =
    require(a.k == b.k, "hammingDistance: hash lengths must match")
    val intersection = intersectionSize(a.active, b.active)
    2 * a.k - 2 * intersection

  def intersectionSize(sortedA: Array[Int], sortedB: Array[Int]): Int =
    var i = 0
    var j = 0
    var count = 0
    while i < sortedA.length && j < sortedB.length do
      val ai = sortedA(i)
      val bj = sortedB(j)
      if ai == bj then
        count += 1
        i += 1
        j += 1
      else if ai < bj then i += 1
      else j += 1
    count

  def fromTopK(topK: Array[Int]): SparseHash =
    SparseHash(topK.clone(), topK.length)
