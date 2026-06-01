package biohash.data

import java.nio.file.Files
import munit.FunSuite

class AnnBenchmarksSuite extends FunSuite:

  test("readFvecs round-trip") {
    val dir = Files.createTempDirectory("biohash-ann")
    try
      val path = dir.resolve("test.fvecs")
      writeFvecs(path, Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
      val vecs = AnnBenchmarks.readFvecs(path)
      assertEquals(vecs.length, 2)
      assertEqualsDouble(vecs(0)(0), 1.0, 1e-6)
      assertEqualsDouble(vecs(1)(1), 4.0, 1e-6)
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
  }

  test("readIvecs round-trip") {
    val dir = Files.createTempDirectory("biohash-ann")
    try
      val path = dir.resolve("test.ivecs")
      writeIvecs(path, Array(Array(10, 20), Array(30, 40)))
      val vecs = AnnBenchmarks.readIvecs(path)
      assertEquals(vecs(0).toSeq, Seq(10, 20))
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
  }

  test("loadSift10K reads siftsmall fvecs subdirectory") {
    val dir = Files.createTempDirectory("biohash-ann")
    try
      val small = Files.createDirectory(dir.resolve("siftsmall"))
      writeFvecs(small.resolve("siftsmall_base.fvecs"), Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
      writeFvecs(small.resolve("siftsmall_query.fvecs"), Array(Array(5.0f, 6.0f)))
      writeIvecs(small.resolve("siftsmall_groundtruth.ivecs"), Array(Array(0, 1)))
      val ann = AnnBenchmarks.loadSift10K(dir)
      assertEquals(ann.name, "sift10k")
      assertEquals(ann.database.length, 2)
      assertEquals(ann.queries.length, 1)
      assertEquals(ann.groundTruth(0), IndexedSeq(0, 1))
      assertEqualsDouble(ann.database(1)(1), 4.0, 1e-6)
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
  }

  private def writeLEInt(out: java.io.OutputStream, value: Int): Unit =
    out.write(value & 0xff)
    out.write((value >>> 8) & 0xff)
    out.write((value >>> 16) & 0xff)
    out.write((value >>> 24) & 0xff)

  private def writeLEFloat(out: java.io.OutputStream, value: Float): Unit =
    writeLEInt(out, java.lang.Float.floatToIntBits(value))

  private def writeFvecs(path: java.nio.file.Path, rows: Array[Array[Float]]): Unit =
    val out = new java.io.BufferedOutputStream(Files.newOutputStream(path))
    try
      rows.foreach { row =>
        writeLEInt(out, row.length)
        row.foreach(f => writeLEFloat(out, f))
      }
    finally out.close()

  private def writeIvecs(path: java.nio.file.Path, rows: Array[Array[Int]]): Unit =
    val out = new java.io.BufferedOutputStream(Files.newOutputStream(path))
    try
      rows.foreach { row =>
        writeLEInt(out, row.length)
        row.foreach(i => writeLEInt(out, i))
      }
    finally out.close()
