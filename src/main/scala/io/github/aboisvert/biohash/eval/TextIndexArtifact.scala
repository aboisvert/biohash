// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash.eval

import io.github.aboisvert.biohash.*
import java.io.{
  BufferedInputStream,
  BufferedOutputStream,
  DataInputStream,
  DataOutputStream,
  FileInputStream,
  FileOutputStream
}
import java.nio.file.{Files, Path}
import java.time.Instant
import java.util.Properties
import scala.jdk.CollectionConverters.*

/** Metadata for one persisted index segment (encoder snapshot + corpus shard). */
final case class TextIndexSegmentManifest(
    segmentId: Int,
    encoderFile: String,
    hashesFile: String,
    idsFile: String,
    itemCount: Int,
    trainingSteps: Long,
    createdAt: String
)

/** One loaded index segment: encoder snapshot and its corpus ids/hashes. */
final case class TextIndexSegment(
    manifest: TextIndexSegmentManifest,
    encoder: HashEncoder,
    corpusIds: IndexedSeq[String],
    corpusHashes: IndexedSeq[SparseHash]
):
  def segmentId: Int = manifest.segmentId

/** Persisted text benchmark index with one or more ordered segments. */
final case class TextIndexArtifact(
    manifest: TextBenchmarkManifest,
    segments: IndexedSeq[TextIndexSegment]
):
  require(segments.nonEmpty, "TextIndexArtifact: at least one segment required")

  def latestSegment: TextIndexSegment = segments.last

  /** Latest encoder snapshot (same as [[latestSegment]] encoder). */
  def encoder: HashEncoder = latestSegment.encoder

  /** All corpus ids across segments in segment order. */
  def corpusIds: IndexedSeq[String] = segments.flatMap(_.corpusIds)

  /** All corpus hashes across segments in segment order. */
  def corpusHashes: IndexedSeq[SparseHash] = segments.flatMap(_.corpusHashes)

  def segmentCount: Int = segments.length

final case class TextBenchmarkManifest(
    dataset: String,
    method: String,
    inputDim: Int,
    k: Int,
    m: Int,
    activity: Double,
    epochs: Int,
    learningRate: Double,
    delta: Double,
    antiWinnerRank: Int,
    seed: Long,
    normalizeInputs: Boolean,
    embeddingModel: String,
    corpusSize: Int,
    querySize: Int,
    trainSeconds: Double,
    encodeSeconds: Double,
    createdAt: String,
    segmentCount: Int = 1,
    totalTrainingSteps: Long = 0L
)

object TextIndexArtifact:

  val ManifestFile = "manifest.properties"
  val SegmentsFile = "segments.properties"
  val EncoderFile = "encoder.bin"
  val CorpusHashesFile = "corpus.hashes.bin"
  val CorpusIdsFile = "corpus.ids"

  def artifactDir(root: Path, dataset: String, config: EvalConfig): Path =
    val tag =
      f"${config.method.toString.toLowerCase}-k${config.k}-a${config.activity}%.4f-ep${config.epochs}-seed${config.seed}"
    root.resolve(dataset).resolve(tag)

  def encoderFileFor(segmentId: Int): String =
    if segmentId == 0 then EncoderFile else s"encoder-$segmentId.bin"

  def hashesFileFor(segmentId: Int): String =
    if segmentId == 0 then CorpusHashesFile else s"corpus-$segmentId.hashes.bin"

  def idsFileFor(segmentId: Int): String =
    if segmentId == 0 then CorpusIdsFile else s"corpus-$segmentId.ids"

  def save(
      dir: Path,
      manifest: TextBenchmarkManifest,
      encoder: HashEncoder,
      corpusIds: IndexedSeq[String],
      corpusHashes: IndexedSeq[SparseHash]
  ): Unit =
    Files.createDirectories(dir)
    writeManifest(dir.resolve(ManifestFile), manifest)
    writeEncoder(dir.resolve(EncoderFile), manifest.method, encoder)
    writeCorpusHashes(dir.resolve(CorpusHashesFile), corpusHashes)
    Files.write(dir.resolve(CorpusIdsFile), corpusIds.asJava)

  def load(dir: Path): TextIndexArtifact =
    require(Files.isDirectory(dir), s"Artifact directory not found: $dir")
    val manifest = readManifest(dir.resolve(ManifestFile))
    val segmentManifests =
      readSegmentIndex(dir).getOrElse(IndexedSeq(legacySegmentManifest(dir, manifest)))
    val segments = segmentManifests.map(loadSegment(dir, manifest, _))
    TextIndexArtifact(manifest, segments)

  /** Append new ids/hashes to the latest segment using its frozen encoder (no new encoder file). */
  def extendLatestSegment(
      dir: Path,
      newIds: IndexedSeq[String],
      newHashes: IndexedSeq[SparseHash]
  ): TextIndexArtifact =
    require(newIds.length == newHashes.length, "extendLatestSegment: ids and hashes length mismatch")
    val manifest = readManifest(dir.resolve(ManifestFile))
    val segmentManifests =
      readSegmentIndex(dir).getOrElse(IndexedSeq(legacySegmentManifest(dir, manifest)))
    val latest = segmentManifests.last
    val existingIds = TextBenchmarkData.readIds(dir.resolve(latest.idsFile))
    val existingHashes = readCorpusHashes(dir.resolve(latest.hashesFile))
    val mergedIds = existingIds ++ newIds
    val mergedHashes = existingHashes ++ newHashes
    writeCorpusHashes(dir.resolve(latest.hashesFile), mergedHashes)
    Files.write(dir.resolve(latest.idsFile), mergedIds.asJava)
    val updatedLatest = latest.copy(itemCount = mergedIds.length)
    val updatedSegments =
      if segmentManifests.length == 1 then IndexedSeq(updatedLatest)
      else segmentManifests.init :+ updatedLatest
    if updatedSegments.length > 1 || Files.exists(dir.resolve(SegmentsFile)) then
      writeSegmentIndex(dir.resolve(SegmentsFile), updatedSegments)
    val updatedManifest = manifest.copy(
      corpusSize = segmentManifests.init.map(_.itemCount).sum + mergedIds.length,
      segmentCount = updatedSegments.length
    )
    writeManifest(dir.resolve(ManifestFile), updatedManifest)
    load(dir)

  /** Write a new segment with a new encoder snapshot and its corpus shard. */
  def appendSegment(
      dir: Path,
      encoder: HashEncoder,
      newIds: IndexedSeq[String],
      newHashes: IndexedSeq[SparseHash],
      trainingSteps: Long
  ): TextIndexArtifact =
    require(newIds.length == newHashes.length, "appendSegment: ids and hashes length mismatch")
    val manifest = readManifest(dir.resolve(ManifestFile))
    val existingSegments =
      readSegmentIndex(dir).getOrElse(IndexedSeq(legacySegmentManifest(dir, manifest)))
    val newSegmentId = existingSegments.last.segmentId + 1
    val createdAt = Instant.now().toString
    val segmentManifest = TextIndexSegmentManifest(
      segmentId = newSegmentId,
      encoderFile = encoderFileFor(newSegmentId),
      hashesFile = hashesFileFor(newSegmentId),
      idsFile = idsFileFor(newSegmentId),
      itemCount = newIds.length,
      trainingSteps = trainingSteps,
      createdAt = createdAt
    )
    writeEncoder(dir.resolve(segmentManifest.encoderFile), manifest.method, encoder)
    writeCorpusHashes(dir.resolve(segmentManifest.hashesFile), newHashes)
    Files.write(dir.resolve(segmentManifest.idsFile), newIds.asJava)
    val allSegments =
      if existingSegments.length == 1 && !Files.exists(dir.resolve(SegmentsFile)) then
        IndexedSeq(existingSegments.head, segmentManifest)
      else existingSegments :+ segmentManifest
    writeSegmentIndex(dir.resolve(SegmentsFile), allSegments)
    val totalCorpusSize = allSegments.map(_.itemCount).sum
    val updatedManifest = manifest.copy(
      corpusSize = totalCorpusSize,
      segmentCount = allSegments.length,
      totalTrainingSteps = trainingSteps
    )
    writeManifest(dir.resolve(ManifestFile), updatedManifest)
    load(dir)

  /** Replace the entire artifact with a single consolidated segment. */
  def replaceWithSingleSegment(
      dir: Path,
      encoder: HashEncoder,
      corpusIds: IndexedSeq[String],
      corpusHashes: IndexedSeq[SparseHash],
      trainingSteps: Long
  ): TextIndexArtifact =
    require(corpusIds.length == corpusHashes.length, "replaceWithSingleSegment: ids and hashes length mismatch")
    val manifest = readManifest(dir.resolve(ManifestFile))
    deleteMultiSegmentFiles(dir)
    writeEncoder(dir.resolve(EncoderFile), manifest.method, encoder)
    writeCorpusHashes(dir.resolve(CorpusHashesFile), corpusHashes)
    Files.write(dir.resolve(CorpusIdsFile), corpusIds.asJava)
    Files.deleteIfExists(dir.resolve(SegmentsFile))
    val updatedManifest = manifest.copy(
      corpusSize = corpusIds.length,
      segmentCount = 1,
      totalTrainingSteps = trainingSteps
    )
    writeManifest(dir.resolve(ManifestFile), updatedManifest)
    load(dir)

  private object TextBenchmarkData:
    def readIds(path: Path): IndexedSeq[String] =
      io.github.aboisvert.biohash.data.TextBenchmark.readIds(path)

  def directorySizeBytes(dir: Path): Long =
    if !Files.isDirectory(dir) then 0L
    else Files.walk(dir).iterator().asScala.filter(Files.isRegularFile(_)).map(p => Files.size(p)).sum

  private def legacySegmentManifest(dir: Path, manifest: TextBenchmarkManifest): TextIndexSegmentManifest =
    val ids = TextBenchmarkData.readIds(dir.resolve(CorpusIdsFile))
    TextIndexSegmentManifest(
      segmentId = 0,
      encoderFile = EncoderFile,
      hashesFile = CorpusHashesFile,
      idsFile = CorpusIdsFile,
      itemCount = ids.length,
      trainingSteps = manifest.totalTrainingSteps,
      createdAt = manifest.createdAt
    )

  private def loadSegment(
      dir: Path,
      manifest: TextBenchmarkManifest,
      segmentManifest: TextIndexSegmentManifest
  ): TextIndexSegment =
    val encoder = readEncoder(dir.resolve(segmentManifest.encoderFile), manifest, segmentManifest.trainingSteps)
    val corpusHashes = readCorpusHashes(dir.resolve(segmentManifest.hashesFile))
    val corpusIds = TextBenchmarkData.readIds(dir.resolve(segmentManifest.idsFile))
    require(corpusIds.length == corpusHashes.length, s"${segmentManifest.idsFile} must match ${segmentManifest.hashesFile}")
    require(corpusIds.length == segmentManifest.itemCount, s"segment ${segmentManifest.segmentId} itemCount mismatch")
    TextIndexSegment(segmentManifest, encoder, corpusIds, corpusHashes)

  private def readSegmentIndex(dir: Path): Option[IndexedSeq[TextIndexSegmentManifest]] =
    val path = dir.resolve(SegmentsFile)
    if !Files.exists(path) then None
    else Some(parseSegmentIndex(path))

  private def parseSegmentIndex(path: Path): IndexedSeq[TextIndexSegmentManifest] =
    val props = new Properties()
    val in = new BufferedInputStream(new FileInputStream(path.toFile))
    try props.load(in)
    finally in.close()
    val count = props.getProperty("segmentCount").toInt
    (0 until count).map { i =>
      TextIndexSegmentManifest(
        segmentId = props.getProperty(s"segment.$i.id").toInt,
        encoderFile = props.getProperty(s"segment.$i.encoderFile"),
        hashesFile = props.getProperty(s"segment.$i.hashesFile"),
        idsFile = props.getProperty(s"segment.$i.idsFile"),
        itemCount = props.getProperty(s"segment.$i.itemCount").toInt,
        trainingSteps = props.getProperty(s"segment.$i.trainingSteps").toLong,
        createdAt = props.getProperty(s"segment.$i.createdAt")
      )
    }.toIndexedSeq

  private def writeSegmentIndex(path: Path, segments: IndexedSeq[TextIndexSegmentManifest]): Unit =
    val props = new Properties()
    props.setProperty("segmentCount", segments.length.toString)
    segments.zipWithIndex.foreach { case (segment, idx) =>
      props.setProperty(s"segment.$idx.id", segment.segmentId.toString)
      props.setProperty(s"segment.$idx.encoderFile", segment.encoderFile)
      props.setProperty(s"segment.$idx.hashesFile", segment.hashesFile)
      props.setProperty(s"segment.$idx.idsFile", segment.idsFile)
      props.setProperty(s"segment.$idx.itemCount", segment.itemCount.toString)
      props.setProperty(s"segment.$idx.trainingSteps", segment.trainingSteps.toString)
      props.setProperty(s"segment.$idx.createdAt", segment.createdAt)
    }
    val out = new BufferedOutputStream(new FileOutputStream(path.toFile))
    try props.store(out, "BioHash text index segments")
    finally out.close()

  private def deleteMultiSegmentFiles(dir: Path): Unit =
    readSegmentIndex(dir).foreach { segments =>
      segments.foreach { segment =>
        if segment.segmentId != 0 then
          Files.deleteIfExists(dir.resolve(segment.encoderFile))
          Files.deleteIfExists(dir.resolve(segment.hashesFile))
          Files.deleteIfExists(dir.resolve(segment.idsFile))
      }
    }

  private def writeManifest(path: Path, manifest: TextBenchmarkManifest): Unit =
    val props = new Properties()
    props.setProperty("dataset", manifest.dataset)
    props.setProperty("method", manifest.method)
    props.setProperty("inputDim", manifest.inputDim.toString)
    props.setProperty("k", manifest.k.toString)
    props.setProperty("m", manifest.m.toString)
    props.setProperty("activity", manifest.activity.toString)
    props.setProperty("epochs", manifest.epochs.toString)
    props.setProperty("learningRate", manifest.learningRate.toString)
    props.setProperty("delta", manifest.delta.toString)
    props.setProperty("antiWinnerRank", manifest.antiWinnerRank.toString)
    props.setProperty("seed", manifest.seed.toString)
    props.setProperty("normalizeInputs", manifest.normalizeInputs.toString)
    props.setProperty("embeddingModel", manifest.embeddingModel)
    props.setProperty("corpusSize", manifest.corpusSize.toString)
    props.setProperty("querySize", manifest.querySize.toString)
    props.setProperty("trainSeconds", manifest.trainSeconds.toString)
    props.setProperty("encodeSeconds", manifest.encodeSeconds.toString)
    props.setProperty("createdAt", manifest.createdAt)
    props.setProperty("segmentCount", manifest.segmentCount.toString)
    props.setProperty("totalTrainingSteps", manifest.totalTrainingSteps.toString)
    val out = new BufferedOutputStream(new FileOutputStream(path.toFile))
    try props.store(out, "BioHash text benchmark artifact")
    finally out.close()

  private def readManifest(path: Path): TextBenchmarkManifest =
    val props = new Properties()
    val in = new BufferedInputStream(new FileInputStream(path.toFile))
    try props.load(in)
    finally in.close()
    def get(key: String): String = props.getProperty(key)
    def getOpt(key: String): Option[String] = Option(props.getProperty(key))

    TextBenchmarkManifest(
      dataset = get("dataset"),
      method = get("method"),
      inputDim = get("inputDim").toInt,
      k = get("k").toInt,
      m = get("m").toInt,
      activity = get("activity").toDouble,
      epochs = get("epochs").toInt,
      learningRate = get("learningRate").toDouble,
      delta = get("delta").toDouble,
      antiWinnerRank = get("antiWinnerRank").toInt,
      seed = get("seed").toLong,
      normalizeInputs = get("normalizeInputs").toBoolean,
      embeddingModel = get("embeddingModel"),
      corpusSize = get("corpusSize").toInt,
      querySize = get("querySize").toInt,
      trainSeconds = get("trainSeconds").toDouble,
      encodeSeconds = get("encodeSeconds").toDouble,
      createdAt = get("createdAt"),
      segmentCount = getOpt("segmentCount").map(_.toInt).getOrElse(1),
      totalTrainingSteps = getOpt("totalTrainingSteps").map(_.toLong).getOrElse(0L)
    )

  private def writeEncoder(path: Path, method: String, encoder: HashEncoder): Unit =
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path.toFile)))
    try
      out.writeUTF(method)
      method.toLowerCase match
        case "biohash" =>
          val bh = encoder.asInstanceOf[BioHash]
          writeBioHashWeights(out, bh.weights)
        case "flyhash" =>
          val fh = encoder.asInstanceOf[FlyHash]
          out.writeDouble(fh.config.samplingRate)
          writeBioHashWeights(out, fh.weights)
        case "naivebiohash" =>
          val nb = encoder.asInstanceOf[NaiveBioHash]
          writeBioHashWeights(out, nb.savedWeights)
        case other => throw IllegalArgumentException(s"Unsupported encoder method: $other")
    finally out.close()

  private def readEncoder(path: Path, manifest: TextBenchmarkManifest): HashEncoder =
    readEncoder(path, manifest, manifest.totalTrainingSteps)

  private def readEncoder(path: Path, manifest: TextBenchmarkManifest, trainingSteps: Long): HashEncoder =
    val in = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val method = in.readUTF()
      require(method.equalsIgnoreCase(manifest.method), s"Encoder method mismatch: $method vs ${manifest.method}")
      method.toLowerCase match
        case "biohash" =>
          val weights = readBioHashWeights(in, manifest.m, manifest.inputDim)
          val config = bioHashConfigFromManifest(manifest)
          BioHash.fromWeights(config, weights, trainingSteps)
        case "flyhash" =>
          val samplingRate = in.readDouble()
          val weights = readBioHashWeights(in, manifest.m, manifest.inputDim)
          val config = FlyHashConfig(
            inputDim = manifest.inputDim,
            m = manifest.m,
            k = manifest.k,
            samplingRate = samplingRate,
            seed = manifest.seed,
            normalizeInputs = manifest.normalizeInputs
          )
          FlyHash.fromWeights(config, weights)
        case "naivebiohash" =>
          val weights = readBioHashWeights(in, manifest.k, manifest.inputDim)
          val config = NaiveBioHashConfig.paper(
            inputDim = manifest.inputDim,
            k = manifest.k,
            learningRate = manifest.learningRate,
            epochs = manifest.epochs,
            antiWinnerRank = manifest.antiWinnerRank,
            delta = manifest.delta,
            seed = manifest.seed,
            normalizeInputs = manifest.normalizeInputs
          )
          NaiveBioHash.fromWeights(config, weights)
        case other => throw IllegalArgumentException(s"Unsupported encoder method: $other")
    finally in.close()

  private def bioHashConfigFromManifest(manifest: TextBenchmarkManifest): BioHashConfig =
    BioHashConfig.paper(
      inputDim = manifest.inputDim,
      m = manifest.m,
      k = manifest.k,
      learningRate = manifest.learningRate,
      epochs = manifest.epochs,
      antiWinnerRank = manifest.antiWinnerRank,
      delta = manifest.delta,
      seed = manifest.seed,
      normalizeInputs = manifest.normalizeInputs
    )

  private def writeBioHashWeights(out: DataOutputStream, weights: Array[Array[Double]]): Unit =
    out.writeInt(weights.length)
    out.writeInt(if weights.isEmpty then 0 else weights.head.length)
    weights.foreach { row =>
      row.foreach(out.writeDouble)
    }

  private def readBioHashWeights(in: DataInputStream, expectedRows: Int, expectedCols: Int): Array[Array[Double]] =
    val rows = in.readInt()
    val cols = in.readInt()
    require(rows == expectedRows, s"Expected $expectedRows weight rows, got $rows")
    require(cols == expectedCols, s"Expected $expectedCols weight cols, got $cols")
    Array.tabulate(rows) { _ =>
      Array.tabulate(cols)(_ => in.readDouble())
    }

  private def writeCorpusHashes(path: Path, hashes: IndexedSeq[SparseHash]): Unit =
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path.toFile)))
    try
      out.writeInt(hashes.length)
      out.writeInt(if hashes.isEmpty then 0 else hashes.head.k)
      hashes.foreach { hash =>
        require(hash.k == hash.active.length)
        out.writeInt(hash.k)
        hash.active.foreach(out.writeInt)
      }
    finally out.close()

  private def readCorpusHashes(path: Path): IndexedSeq[SparseHash] =
    val in = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val n = in.readInt()
      val expectedK = in.readInt()
      (0 until n).map { _ =>
        val k = in.readInt()
        require(k == expectedK, s"Inconsistent hash length: expected $expectedK, got $k")
        val active = Array.tabulate(k)(_ => in.readInt())
        SparseHash(active, k)
      }.toIndexedSeq
    finally in.close()
