// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.data

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** CIFAR-10 binary batch loader (32x32x3 flattened to 3072-dim, normalized to [0,1]). */
object Cifar10:

  val DefaultDir = Path.of("data", "cifar-10-batches-bin")

  def load(dir: Path = DefaultDir): LabeledDataset =
    val trainBatches = (1 to 5).flatMap { i =>
      readBatch(dir.resolve(s"data_batch_$i.bin"))
    }
    val testBatch = readBatch(dir.resolve("test_batch.bin"))
    val vectors = (trainBatches ++ testBatch).map(_._1)
    val labels = (trainBatches ++ testBatch).map(_._2)
    LabeledDataset(vectors, labels, "cifar-10")

  private def readBatch(path: Path): IndexedSeq[(Array[Double], Int)] =
    val bytes = Files.readAllBytes(path)
    val numRecords = 10000
    val recordSize = 1 + 3072
    require(bytes.length == numRecords * recordSize)
    (0 until numRecords).map { r =>
      val offset = r * recordSize
      val label = bytes(offset) & 0xff
      val pixels = new Array[Double](3072)
      var i = 0
      while i < 3072 do
        pixels(i) = (bytes(offset + 1 + i) & 0xff).toDouble / 255.0
        i += 1
      (pixels, label)
    }.toIndexedSeq

  def paperSplit(seed: Long = 42L): RetrievalSplit =
    DatasetSplit.byClass(load(), queriesPerClass = 1000, seed = seed)

  def isAvailable(dir: Path = DefaultDir): Boolean =
    Files.exists(dir.resolve("data_batch_1.bin")) && Files.exists(dir.resolve("test_batch.bin"))

  /** Load pre-extracted VGG16 fc7 features if available (CSV or binary). */
  def loadVggFc7Features(path: Path): LabeledDataset =
    if path.toString.endsWith(".csv") then loadCsvFeatures(path)
    else loadBinaryFeatures(path)

  private def loadCsvFeatures(path: Path): LabeledDataset =
    val lines = Files.readAllLines(path)
    val rows = lines.iterator().asScala.drop(1).map(_.split(",")).toIndexedSeq
    val labels = rows.map(r => r.last.toInt)
    val vectors = rows.map { r =>
      r.dropRight(1).map(_.toDouble)
    }
    LabeledDataset(vectors, labels, "cifar10-vgg-fc7")

  private def loadBinaryFeatures(path: Path): LabeledDataset =
    val stream = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val n = stream.readInt()
      val d = stream.readInt()
      val vectors = (0 until n).map { _ =>
        val v = new Array[Double](d)
        var i = 0
        while i < d do
          v(i) = stream.readDouble()
          i += 1
        v
      }
      val labels = (0 until n).map(_ => stream.readInt())
      LabeledDataset(vectors, labels, "cifar10-vgg-fc7")
    finally stream.close()
