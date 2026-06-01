package biohash.eval

import biohash.*
import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, FileInputStream, FileOutputStream}
import java.nio.file.{Files, Path}
import java.time.Instant
import java.util.Properties
import scala.jdk.CollectionConverters.*

/** Persisted text benchmark index: trained encoder weights and corpus sparse hashes. */
final case class TextIndexArtifact(
    manifest: TextBenchmarkManifest,
    encoder: HashEncoder,
    corpusIds: IndexedSeq[String],
    corpusHashes: IndexedSeq[SparseHash]
)

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
    createdAt: String
)

object TextIndexArtifact:

  val ManifestFile = "manifest.properties"
  val EncoderFile = "encoder.bin"
  val CorpusHashesFile = "corpus.hashes.bin"
  val CorpusIdsFile = "corpus.ids"

  def artifactDir(root: Path, dataset: String, config: EvalConfig): Path =
    val tag =
      f"${config.method.toString.toLowerCase}-k${config.k}-a${config.activity}%.4f-ep${config.epochs}-seed${config.seed}"
    root.resolve(dataset).resolve(tag)

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
    val encoder = readEncoder(dir.resolve(EncoderFile), manifest)
    val corpusHashes = readCorpusHashes(dir.resolve(CorpusHashesFile))
    val corpusIds = TextBenchmarkData.readIds(dir.resolve(CorpusIdsFile))
    require(corpusIds.length == corpusHashes.length, "corpus.ids must match corpus.hashes.bin")
    TextIndexArtifact(manifest, encoder, corpusIds, corpusHashes)

  private object TextBenchmarkData:
    def readIds(path: Path): IndexedSeq[String] =
      biohash.data.TextBenchmark.readIds(path)

  def directorySizeBytes(dir: Path): Long =
    if !Files.isDirectory(dir) then 0L
    else
      Files.walk(dir).iterator().asScala.filter(Files.isRegularFile(_)).map(p => Files.size(p)).sum

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
    val out = new BufferedOutputStream(new FileOutputStream(path.toFile))
    try props.store(out, "BioHash text benchmark artifact")
    finally out.close()

  private def readManifest(path: Path): TextBenchmarkManifest =
    val props = new Properties()
    val in = new BufferedInputStream(new FileInputStream(path.toFile))
    try props.load(in)
    finally in.close()
    def get(key: String): String = props.getProperty(key)

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
      createdAt = get("createdAt")
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
    val in = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile)))
    try
      val method = in.readUTF()
      require(method.equalsIgnoreCase(manifest.method), s"Encoder method mismatch: $method vs ${manifest.method}")
      method.toLowerCase match
        case "biohash" =>
          val weights = readBioHashWeights(in, manifest.m, manifest.inputDim)
          val config = BioHashConfig.paper(
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
          BioHash.fromWeights(config, weights)
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
