package biohash.data

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.file.{Files, Path}

/** TEXMEX ANN benchmark format: .bvecs (uint8) and .fvecs (float32) vectors (little-endian). */
object AnnBenchmarks:

  private def readLEInt(stream: java.io.InputStream): Int =
    val b0 = stream.read()
    val b1 = stream.read()
    val b2 = stream.read()
    val b3 = stream.read()
    if b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0 then throw new java.io.EOFException()
    (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24)

  private def readLEFloat(stream: java.io.InputStream): Double =
    java.lang.Float.intBitsToFloat(readLEInt(stream)).toDouble

  private def readDim(stream: java.io.InputStream): Option[Int] =
    try Some(readLEInt(stream))
    catch case _: java.io.EOFException => None

  def readBvecs(path: Path, maxRecords: Option[Int] = None): IndexedSeq[Array[Double]] =
    readVecs(path, maxRecords) { stream =>
      stream.readUnsignedByte().toDouble
    }

  def readFvecs(path: Path, maxRecords: Option[Int] = None): IndexedSeq[Array[Double]] =
    readVecs(path, maxRecords)(readLEFloat)

  def readIvecs(path: Path, maxRecords: Option[Int] = None): IndexedSeq[Array[Int]] =
    val stream = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val result = scala.collection.mutable.ArrayBuffer.empty[Array[Int]]
      var count = 0
      var continue = true
      while continue && maxRecords.forall(count < _) do
        readDim(stream) match
          case None => continue = false
          case Some(dim) =>
            val row = new Array[Int](dim)
            var i = 0
            while i < dim do
              row(i) = readLEInt(stream)
              i += 1
            result += row
            count += 1
      result.toIndexedSeq
    finally stream.close()

  private def readVecs(
      path: Path,
      maxRecords: Option[Int]
  )(readElement: DataInputStream => Double): IndexedSeq[Array[Double]] =
    val stream = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val result = scala.collection.mutable.ArrayBuffer.empty[Array[Double]]
      var count = 0
      var continue = true
      while continue && maxRecords.forall(count < _) do
        readDim(stream) match
          case None => continue = false
          case Some(dim) =>
            val row = new Array[Double](dim)
            var i = 0
            while i < dim do
              row(i) = readElement(stream)
              i += 1
            result += row
            count += 1
      result.toIndexedSeq
    finally stream.close()

  final case class AnnDataset(
      database: IndexedSeq[Array[Double]],
      queries: IndexedSeq[Array[Double]],
      groundTruth: IndexedSeq[IndexedSeq[Int]],
      name: String
  )

  def loadSift1M(dir: Path, maxDb: Option[Int] = None, maxQueries: Option[Int] = None): AnnDataset =
    AnnDataset(
      database = readBvecs(dir.resolve("sift_base.bvecs"), maxDb),
      queries = readBvecs(dir.resolve("sift_query.bvecs"), maxQueries),
      groundTruth = readIvecs(dir.resolve("sift_groundtruth.ivecs"), maxQueries).map(_.toIndexedSeq),
      name = "sift1m"
    )

  def loadSift10K(dir: Path): AnnDataset =
    val small = dir.resolve("siftsmall")
    AnnDataset(
      database = readFvecs(small.resolve("siftsmall_base.fvecs")),
      queries = readFvecs(small.resolve("siftsmall_query.fvecs")),
      groundTruth =
        readIvecs(small.resolve("siftsmall_groundtruth.ivecs")).map(_.toIndexedSeq),
      name = "sift10k"
    )

  def loadGist1M(dir: Path, maxDb: Option[Int] = None, maxQueries: Option[Int] = None): AnnDataset =
    AnnDataset(
      database = readFvecs(dir.resolve("gist_base.fvecs"), maxDb),
      queries = readFvecs(dir.resolve("gist_query.fvecs"), maxQueries),
      groundTruth = readIvecs(dir.resolve("gist_groundtruth.ivecs"), maxQueries).map(_.toIndexedSeq),
      name = "gist1m"
    )

  def isAvailable(dir: Path, prefix: String): Boolean =
    Files.exists(dir.resolve(s"${prefix}_base.bvecs")) ||
      Files.exists(dir.resolve(s"${prefix}_base.fvecs"))
