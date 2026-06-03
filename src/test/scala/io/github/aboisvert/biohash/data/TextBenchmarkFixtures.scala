// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.data

import java.io.{BufferedOutputStream, DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Path}

object TextBenchmarkFixtures:

  def writeFixture(root: Path): Path =
    val dir = root.resolve("mini")
    Files.createDirectories(dir.resolve("qrels"))

    val corpusIds = IndexedSeq("d1", "d2", "d3", "d4")
    val queryIds = IndexedSeq("q1", "q2")
    val corpusVectors = IndexedSeq(
      Array(1.0, 0.0, 0.0),
      Array(0.9, 0.1, 0.0),
      Array(0.0, 1.0, 0.0),
      Array(0.0, 0.0, 1.0)
    )
    val queryVectors = IndexedSeq(
      Array(0.95, 0.05, 0.0),
      Array(0.05, 0.95, 0.0)
    )

    writeIds(dir.resolve("corpus.ids"), corpusIds)
    writeIds(dir.resolve("query.ids"), queryIds)
    writeFvecs(dir.resolve("corpus.fvecs"), corpusVectors)
    writeFvecs(dir.resolve("query.fvecs"), queryVectors)
    Files.writeString(
      dir.resolve("manifest.properties"),
      "embeddingModel=test-fixture\ncorpusSize=4\nquerySize=2\nvectorDim=3\n"
    )
    Files.writeString(
      dir.resolve("qrels").resolve("test.tsv"),
      "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td3\t1\n"
    )
    Files.writeString(
      dir.resolve("corpus.jsonl"),
      """{"_id":"d1","title":"A","text":"alpha"}
        |{"_id":"d2","title":"B","text":"almost alpha"}
        |{"_id":"d3","title":"C","text":"beta"}
        |{"_id":"d4","title":"D","text":"gamma"}""".stripMargin
    )
    Files.writeString(
      dir.resolve("queries.jsonl"),
      """{"_id":"q1","text":"find alpha topic"}
        |{"_id":"q2","text":"find beta topic"}""".stripMargin
    )
    dir

  def writeIds(path: Path, ids: IndexedSeq[String]): Unit =
    Files.write(path, ids.mkString("\n").concat("\n").getBytes)

  def writeFvecs(path: Path, vectors: IndexedSeq[Array[Double]]): Unit =
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path.toFile)))
    try
      vectors.foreach { vector =>
        writeLEInt(out, vector.length)
        vector.foreach(value => writeLEFloat(out, value.toFloat))
      }
    finally out.close()

  private def writeLEInt(out: DataOutputStream, value: Int): Unit =
    out.writeByte(value & 0xff)
    out.writeByte((value >> 8) & 0xff)
    out.writeByte((value >> 16) & 0xff)
    out.writeByte((value >> 24) & 0xff)

  private def writeLEFloat(out: DataOutputStream, value: Float): Unit =
    writeLEInt(out, java.lang.Float.floatToIntBits(value))
