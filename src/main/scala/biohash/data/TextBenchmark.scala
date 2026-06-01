package biohash.data

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** BEIR-style text retrieval benchmark: corpus/query embeddings plus qrels. */
final case class TextBenchmarkDataset(
    name: String,
    corpusIds: IndexedSeq[String],
    corpusVectors: IndexedSeq[Array[Double]],
    queryIds: IndexedSeq[String],
    queryVectors: IndexedSeq[Array[Double]],
    qrels: Map[String, Map[String, Int]],
    embeddingModel: String
):
  def corpusSize: Int = corpusVectors.length
  def querySize: Int = queryVectors.length
  def inputDim: Int = if corpusVectors.isEmpty then 0 else corpusVectors.head.length

  def corpusIdToIndex: Map[String, Int] =
    corpusIds.zipWithIndex.toMap

object TextBenchmark:

  val DefaultRoot = Path.of("data", "text")

  def datasetDir(root: Path, name: String): Path = root.resolve(name)

  def isAvailable(dir: Path): Boolean =
    Files.isDirectory(dir) &&
      Files.exists(dir.resolve("corpus.fvecs")) &&
      Files.exists(dir.resolve("corpus.ids")) &&
      Files.exists(dir.resolve("query.fvecs")) &&
      Files.exists(dir.resolve("query.ids")) &&
      Files.exists(dir.resolve("qrels").resolve("test.tsv"))

  def load(dir: Path, name: Option[String] = None): TextBenchmarkDataset =
    val datasetName = name.getOrElse(dir.getFileName.toString)
    require(isAvailable(dir), s"Text benchmark data not found in $dir")

    val corpusIds = readIds(dir.resolve("corpus.ids"))
    val queryIds = readIds(dir.resolve("query.ids"))
    val corpusVectors = AnnBenchmarks.readFvecs(dir.resolve("corpus.fvecs"))
    val queryVectors = AnnBenchmarks.readFvecs(dir.resolve("query.fvecs"))
    val qrels = readQrels(dir.resolve("qrels").resolve("test.tsv"))
    val embeddingModel = readEmbeddingModel(dir)

    require(
      corpusIds.length == corpusVectors.length,
      s"corpus.ids (${corpusIds.length}) must match corpus.fvecs (${corpusVectors.length})"
    )
    require(
      queryIds.length == queryVectors.length,
      s"query.ids (${queryIds.length}) must match query.fvecs (${queryVectors.length})"
    )
    if corpusVectors.nonEmpty then
      val dim = corpusVectors.head.length
      require(queryVectors.forall(_.length == dim), "query vector dimension must match corpus")
      require(corpusVectors.forall(_.length == dim), "corpus vectors must share one dimension")

    TextBenchmarkDataset(
      name = datasetName,
      corpusIds = corpusIds,
      corpusVectors = corpusVectors,
      queryIds = queryIds,
      queryVectors = queryVectors,
      qrels = qrels,
      embeddingModel = embeddingModel
    )

  /** One external id per line, aligned with `.fvecs` row order. */
  def readIds(path: Path): IndexedSeq[String] =
    Files.readAllLines(path).asScala.map(_.trim).filter(_.nonEmpty).toIndexedSeq

  /** BEIR qrels TSV: query-id, corpus-id, score (header row required). */
  def readQrels(path: Path): Map[String, Map[String, Int]] =
    val lines = Files.readAllLines(path).asScala.toIndexedSeq
    require(lines.nonEmpty, s"qrels file is empty: $path")
    val body = if lines.head.toLowerCase.contains("query") then lines.drop(1) else lines
    val grouped = body.flatMap { line =>
      val parts = line.trim.split("\t")
      if parts.length >= 3 then
        val queryId = parts(0).trim
        val corpusId = parts(1).trim
        val score = parts(2).trim.toInt
        if score > 0 then Some((queryId, corpusId, score)) else None
      else None
    }.groupMap(_._1)(row => row._2 -> row._3)

    grouped.view.mapValues { entries =>
      entries.map { case (docId, score) => docId -> score }.toMap
    }.toMap

  private def readEmbeddingModel(dir: Path): String =
    val manifest = dir.resolve("manifest.properties")
    if Files.exists(manifest) then
      Files.readAllLines(manifest).asScala
        .collectFirst:
          case line if line.startsWith("embeddingModel=") => line.stripPrefix("embeddingModel=").trim
        .getOrElse("unknown")
    else "unknown"

  /** Extract `_id` from a BEIR jsonl line without pulling in a JSON library. */
  def parseJsonlId(line: String): Option[String] =
    val key = "\"_id\""
    val start = line.indexOf(key)
    if start < 0 then None
    else
      val colon = line.indexOf(':', start + key.length)
      if colon < 0 then None
      else
        val quoteStart = line.indexOf('"', colon + 1)
        if quoteStart < 0 then None
        else
          val quoteEnd = line.indexOf('"', quoteStart + 1)
          if quoteEnd < 0 then None else Some(line.substring(quoteStart + 1, quoteEnd))

  def countJsonlRecords(path: Path): Int =
    if Files.exists(path) then Files.readAllLines(path).size else 0
