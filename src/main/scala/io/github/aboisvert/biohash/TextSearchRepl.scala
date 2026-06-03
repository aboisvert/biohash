// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.data.TextBenchmark
import io.github.aboisvert.biohash.eval.{SearchHit, TextIndexArtifact, TextSearchService}
import java.io.{InputStream, OutputStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.concurrent.TimeUnit
import scala.collection.mutable.ArrayBuffer
import scala.io.StdIn.readLine
import scala.util.{Failure, Success, Try, Using}

object TextSearchReplApp:

  private val EmbedScript = Path.of("scripts", "embed_query.py")
  private val ProjectVenvPython = Path.of(".venv", "bin", "python")
  private val SnippetLength = 300
  private val ListQueryLimit = 20
  private val QuitPayload = "__QUIT__"
  private val ReadyPayload = "READY"
  private val ErrPrefix = "ERR\t"
  private val EmbedServerShutdownSeconds = 5

  @main def textSearchRepl(args: String*): Unit =
    val options = Main.parseCli(args)
    Main.requireKnown(
      options,
      Set(
        "dataset",
        "method",
        "k",
        "activity",
        "epochs",
        "learning-rate",
        "learningRate",
        "delta",
        "anti-winner-rank",
        "antiWinnerRank",
        "seed",
        "data-dir",
        "dataDir",
        "artifact-dir",
        "artifactDir",
        "top-k",
        "topK",
        "python",
        "embed-script",
        "embedScript"
      )
    )
    Main.requireMaxPositionals(options, 0)
    val (dataset, dataDir, artifactRoot, config, _, _) = Main.textBenchmarkOptions(options)
    val defaultTopK = options.values
      .get("top-k")
      .orElse(options.values.get("topK"))
      .filter(_.nonEmpty)
      .map(_.toInt)
      .getOrElse(10)
    val python = resolvePython(options.values.get("python"))
    val embedScript = options.values
      .get("embed-script")
      .orElse(options.values.get("embedScript"))
      .map(Path.of(_))
      .getOrElse(EmbedScript)

    if !TextBenchmark.isAvailable(dataDir) then
      println(
        s"Text benchmark data not found in $dataDir.\n" +
          prepareHint(dataset) + "\n" +
          "See data/README.md for the expected layout."
      )
      sys.exit(1)

    val artifactDir = TextIndexArtifact.artifactDir(artifactRoot, dataset, config)
    if !Files.exists(artifactDir.resolve(TextIndexArtifact.ManifestFile)) then
      println(
        s"Trained artifact not found in $artifactDir.\n" +
          s"Run `trainTextBenchmark --dataset $dataset` with matching hyperparameters first."
      )
      sys.exit(1)

    if !Files.exists(embedScript) then
      println(s"Embedding script not found: $embedScript")
      sys.exit(1)

    if !embeddingDepsAvailable(python) then
      println(
        s"Python at $python does not have sentence-transformers.\n" +
          "Run: just install-python-deps\n" +
          "Or:  pip install -r scripts/requirements.txt\n" +
          "Or:  just text-search-repl dataset=" + dataset + " --python /path/to/venv/bin/python"
      )
      sys.exit(1)

    val search = TextSearchService.load(artifactDir)
    val passages = TextBenchmark.loadCorpusPassages(dataDir)
    val queryTexts = TextBenchmark.loadQueryTexts(dataDir)
    val datasetLoaded = TextBenchmark.load(dataDir, Some(dataset))
    val queryIdToVector = datasetLoaded.queryIds.zip(datasetLoaded.queryVectors).toMap
    val manifestPath = artifactDir.resolve(TextIndexArtifact.ManifestFile)

    println(
      s"Text search — dataset=$dataset method=${search.manifest.method} hash-k=${search.manifest.k} corpus=${search.manifest.corpusSize}"
    )
    println(s"Embedding model: ${search.manifest.embeddingModel}")
    println(s"Python: $python")
    println(s"Artifact: $artifactDir")
    println("Enter a query, :help for commands. Top-k defaults to " + defaultTopK + ".")
    println()

    Try(QueryEmbedder.start(python, embedScript, manifestPath)) match
      case Failure(err) =>
        println(s"Failed to start embedding server: ${err.getMessage}")
        println("Run: just install-python-deps")
        println("Or:  pip install -r scripts/requirements.txt")
        sys.exit(1)
      case Success(embedder) =>
        Using.resource(embedder): emb =>
          var topK = defaultTopK
          var running = true
          while running do
            val raw = readLine("query> ")
            if raw == null then running = false
            else
              val line = raw.trim
              if line.isEmpty then ()
              else if line == ":quit" || line == ":q" then running = false
              else if line == ":help" then printHelp()
              else if line.startsWith(":k ") then
                Try(line.drop(3).trim.toInt) match
                  case Success(n) if n > 0 =>
                    topK = n
                    println(s"top-k set to $n")
                  case Success(_) => println(":k requires a positive integer")
                  case Failure(_) => println(":k requires a positive integer")
              else if line == ":list" then
                val ids = TextBenchmark.listQueryIds(dataDir, ListQueryLimit)
                if ids.isEmpty then println("No queries.jsonl found.")
                else ids.foreach(id => println(s"  $id"))
              else if line.startsWith(":use ") then
                val queryId = line.drop(5).trim
                queryIdToVector.get(queryId) match
                  case Some(vector) =>
                    val text = queryTexts.getOrElse(queryId, queryId)
                    println(s"Query [$queryId]: ${truncate(text, SnippetLength)}")
                    runSearch(search, passages, vector, topK)
                  case None => println(s"Unknown query id: $queryId (try :list)")
              else
                emb.embed(line) match
                  case Left(message) => println(message)
                  case Right(vector) => runSearch(search, passages, vector, topK)

        println("Bye.")

  /** Prefer explicit --python, then project .venv, then active virtualenv, else python3. */
  private def resolvePython(explicit: Option[String]): String =
    explicit.filter(_.nonEmpty) match
      case Some(path) => path
      case None =>
        if Files.isExecutable(ProjectVenvPython) then ProjectVenvPython.toString
        else
          sys.env
            .get("VIRTUAL_ENV")
            .map(env => Path.of(env, "bin", "python"))
            .filter(Files.isExecutable)
            .map(_.toString)
            .getOrElse("python3")

  private def embeddingDepsAvailable(python: String): Boolean =
    val pb = new ProcessBuilder(python, "-c", "import sentence_transformers")
    pb.redirectErrorStream(true)
    Try:
      val proc = pb.start()
      proc.getInputStream.close()
      proc.waitFor() == 0
    .getOrElse(false)

  private def prepareHint(dataset: String): String =
    dataset match
      case "gutenberg" =>
        "Run `just prepare-text-gutenberg` or `python scripts/prepare_gutenberg_embeddings.py` first."
      case s if s.startsWith("narrativeqa") =>
        "Run `just prepare-text-narrative` or `python scripts/prepare_narrative_embeddings.py` first."
      case other =>
        s"Run `just prepare-text dataset=$other` or `python scripts/prepare_beir_embeddings.py --dataset $other` first."

  private def printHelp(): Unit =
    println("""Commands:
      |  :help        show this help
      |  :quit, :q    exit
      |  :k N         set number of results (top-k)
      |  :list        list benchmark query ids (if queries.jsonl is present)
      |  :use <id>    search using a pre-embedded benchmark query
      |  <text>       embed query text and search the corpus""".stripMargin)

  private def runSearch(
      search: TextSearchService,
      passages: Map[String, String],
      queryVector: Array[Double],
      topK: Int
  ): Unit =
    val hits = search.search(queryVector, topK)
    if hits.isEmpty then println("(no results)")
    else hits.foreach(hit => printHit(hit, passages.get(hit.docId)))

  private def printHit(hit: SearchHit, passage: Option[String]): Unit =
    val snippet = passage.map(p => truncate(p.replace('\n', ' '), SnippetLength)).getOrElse("(no passage text)")
    println(f"  ${hit.rank}%2d  ${hit.docId}%-28s  hamming=${hit.hamming}%3d  $snippet")

  private def truncate(text: String, maxLen: Int): String =
    if text.length <= maxLen then text else text.take(maxLen - 3) + "..."

  private def writeFrame(out: OutputStream, payload: Array[Byte]): Unit =
    out.write(s"${payload.length}\n".getBytes(UTF_8))
    out.write(payload)
    out.flush()

  private def readLengthLine(in: InputStream): Option[Int] =
    val buf = ArrayBuffer[Byte]()
    var byte = in.read()
    while byte != -1 do
      if byte == '\n'.toInt then
        return Try(new String(buf.toArray, UTF_8).trim.toInt).toOption
      buf += byte.toByte
      byte = in.read()
    None

  private def readFrame(in: InputStream): Option[Array[Byte]] =
    readLengthLine(in).flatMap: length =>
      if length < 0 then None
      else
        val payload = in.readNBytes(length)
        if payload.length == length then Some(payload) else None

  private def parseEmbedResponse(payload: Array[Byte]): Either[String, Array[Double]] =
    val text = new String(payload, UTF_8)
    if text.startsWith(ErrPrefix) then Left(text.stripPrefix(ErrPrefix))
    else
      Try(text.split("\\s+").map(_.toDouble)) match
        case Success(vector) => Right(vector)
        case Failure(parseErr) => Left(s"Failed to parse embedding: ${parseErr.getMessage}")

  private def drainStderr(proc: Process): Thread =
    val thread = Thread(() =>
      val err = proc.getErrorStream
      val buf = new Array[Byte](4096)
      var n = err.read(buf)
      while n >= 0 do
        n = err.read(buf)
    )
    thread.setDaemon(true)
    thread.start()
    thread

  private final class QueryEmbedder private (
      proc: Process,
      out: OutputStream,
      in: InputStream,
      stderrDrainer: Thread
  ) extends AutoCloseable:

    private var closed = false

    def embed(text: String): Either[String, Array[Double]] = synchronized:
      if closed then return Left("Embedding server is closed")
      if !proc.isAlive then return Left("Embedding server exited unexpectedly. Restart text-search-repl.")
      Try:
        writeFrame(out, text.getBytes(UTF_8))
        readFrame(in).map(parseEmbedResponse).getOrElse(Left("Embedding server exited unexpectedly. Restart text-search-repl."))
      match
        case Success(result) => result
        case Failure(err)    => Left(s"Embedding failed: ${err.getMessage}")

    def close(): Unit = synchronized:
      if closed then return
      closed = true
      Try(writeFrame(out, QuitPayload.getBytes(UTF_8)))
      if !proc.waitFor(EmbedServerShutdownSeconds, TimeUnit.SECONDS) then proc.destroyForcibly()
      Try(out.close())
      Try(in.close())
      Try(stderrDrainer.join(EmbedServerShutdownSeconds * 1000L))

  private object QueryEmbedder:
    def start(python: String, embedScript: Path, manifestPath: Path): QueryEmbedder =
      val command = Array(
        python,
        embedScript.toString,
        "--server",
        "--manifest",
        manifestPath.toString
      )
      val pb = new ProcessBuilder(command*)
      pb.directory(Path.of(".").toFile)
      val proc = pb.start()
      val stderrDrainer = drainStderr(proc)
      val embedder = new QueryEmbedder(proc, proc.getOutputStream, proc.getInputStream, stderrDrainer)
      println("Loading embedding model (one-time)...")
      readFrame(proc.getInputStream) match
        case Some(payload) if new String(payload, UTF_8) == ReadyPayload => embedder
        case Some(payload) =>
          embedder.close()
          throw new RuntimeException(s"Unexpected startup response: ${new String(payload, UTF_8)}")
        case None =>
          embedder.close()
          val exit = proc.waitFor()
          throw new RuntimeException(s"Embed server exited during startup (code $exit)")
