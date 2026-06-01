// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash.data

import java.nio.file.Files
import munit.FunSuite

class IdxReaderSuite extends FunSuite:

  test("readImages and readLabels from generated IDX files") {
    val dir = Files.createTempDirectory("biohash-idx")
    try
      writeMiniIdxImages(dir.resolve("images-idx3-ubyte"), rows = 2, cols = 2, pixels = Array(
        Array(0, 255, 128, 64),
        Array(10, 20, 30, 40)
      ))
      writeMiniIdxLabels(dir.resolve("labels-idx1-ubyte"), labels = Array(3, 7))

      val images = IdxReader.readImages(dir.resolve("images-idx3-ubyte"))
      assertEquals(images.length, 2)
      assertEqualsDouble(images(0)(0), 0.0, 1e-9)
      assertEqualsDouble(images(0)(1), 1.0, 1e-9)

      val lbls = IdxReader.readLabels(dir.resolve("labels-idx1-ubyte"))
      assertEquals(lbls.toSeq, Seq(3, 7))
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
  }

  private def writeMiniIdxImages(path: java.nio.file.Path, rows: Int, cols: Int, pixels: Array[Array[Int]]): Unit =
    val out = new java.io.DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(path)))
    try
      out.writeInt(2051)
      out.writeInt(pixels.length)
      out.writeInt(rows)
      out.writeInt(cols)
      pixels.foreach { row =>
        row.foreach(b => out.writeByte(b))
      }
    finally out.close()

  private def writeMiniIdxLabels(path: java.nio.file.Path, labels: Array[Int]): Unit =
    val out = new java.io.DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(path)))
    try
      out.writeInt(2049)
      out.writeInt(labels.length)
      labels.foreach(b => out.writeByte(b))
    finally out.close()
