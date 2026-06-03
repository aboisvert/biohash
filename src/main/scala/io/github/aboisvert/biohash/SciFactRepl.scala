// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.data.TextBenchmark
import io.github.aboisvert.biohash.eval.{SearchHit, TextIndexArtifact, TextSearchService}
import java.nio.file.{Files, Path}
import scala.io.StdIn.readLine
import scala.util.{Failure, Success, Try}

object SciFactReplApp:

  private val EmbedScript = Path.of("scripts", "embed_query.py")
  private val ProjectVenvPython = Path.of(".venv", "bin", "python")
  private val SnippetLength = 200
  private val ListQueryLimit = 20

  @main def scifactRepl(args: String*): Unit =
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
          s"Run `python scripts/prepare_beir_embeddings.py --dataset $dataset` first.\n" +
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
          "Or:  just scifact-repl --python /path/to/venv/bin/python"
      )
      sys.exit(1)

    val search = TextSearchService.load(artifactDir)
    val passages = TextBenchmark.loadCorpusPassages(dataDir)
    val queryTexts = TextBenchmark.loadQueryTexts(dataDir)
    val datasetLoaded = TextBenchmark.load(dataDir, Some(dataset))
    val queryIdToVector = datasetLoaded.queryIds.zip(datasetLoaded.queryVectors).toMap
    val manifestPath = artifactDir.resolve(TextIndexArtifact.ManifestFile)

    println(
      s"SciFact search — dataset=$dataset method=${search.manifest.method} hash-k=${search.manifest.k} corpus=${search.manifest.corpusSize}"
    )
    println(s"Embedding model: ${search.manifest.embeddingModel}")
    println(s"Python: $python")
    println(s"Artifact: $artifactDir")
    println("Enter a query, :help for commands. Top-k defaults to " + defaultTopK + ".")
    println()

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
          embedQuery(python, embedScript, manifestPath, line) match
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

  private def printHelp(): Unit =
    println("""Commands:
      |  :help        show this help
      |  :quit, :q    exit
      |  :k N         set number of results (top-k)
      |  :list        list benchmark query ids
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
    println(f"  ${hit.rank}%2d  ${hit.docId}%-12s  hamming=${hit.hamming}%3d  $snippet")

  private def truncate(text: String, maxLen: Int): String =
    if text.length <= maxLen then text else text.take(maxLen - 3) + "..."

  private def embedQuery(
      python: String,
      embedScript: Path,
      manifestPath: Path,
      text: String
  ): Either[String, Array[Double]] =
    val command = Array(
      python,
      embedScript.toString,
      "--text",
      text,
      "--manifest",
      manifestPath.toString
    )
    val pb = new ProcessBuilder(command*)
    pb.directory(Path.of(".").toFile)
    pb.redirectErrorStream(false)
    Try:
      val proc = pb.start()
      val output =
        try scala.io.Source.fromInputStream(proc.getInputStream).mkString
        finally proc.getInputStream.close()
      val exit = proc.waitFor()
      if exit != 0 then throw new RuntimeException(s"embed script (${command.mkString(" ")}) exited with code $exit: $output")
      output
    match
      case Failure(err) =>
        Left(
          s"Embedding failed: ${err.getMessage}\n" +
            "Run: just install-python-deps\n" +
            "Or:  pip install -r scripts/requirements.txt"
        )
      case Success(output) =>
        val line = output.trim.linesIterator.nextOption().getOrElse("")
        if line.isEmpty then Left("Embedding produced no output")
        else
          Try(line.split("\\s+").map(_.toDouble)) match
            case Success(vector) => Right(vector)
            case Failure(parseErr) =>
              Left(s"Failed to parse embedding: ${parseErr.getMessage}")
