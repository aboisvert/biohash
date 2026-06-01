// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import io.github.aboisvert.biohash.data.*
import io.github.aboisvert.biohash.eval.*
import java.nio.file.{Files, Path}

object Main:

  private final case class CliOptions(values: Map[String, String], positionals: Vector[String])

  private def parseCli(args: Seq[String]): CliOptions =
    var values = Map.empty[String, String]
    var positionals = Vector.empty[String]
    var i = 0
    while i < args.length do
      val arg = args(i)
      if arg.startsWith("--") then
        val option = arg.drop(2)
        val equalsAt = option.indexOf('=')
        if option.isEmpty then throw IllegalArgumentException("Empty option name")
        if equalsAt >= 0 then
          val name = option.take(equalsAt)
          val value = option.drop(equalsAt + 1)
          if name.isEmpty then throw IllegalArgumentException(s"Invalid option: $arg")
          values += name -> value
          i += 1
        else
          if i + 1 >= args.length || args(i + 1).startsWith("--") then
            throw IllegalArgumentException(s"Missing value for option: $arg")
          values += option -> args(i + 1)
          i += 2
      else
        positionals :+= arg
        i += 1
    CliOptions(values, positionals)

  private def requireKnown(options: CliOptions, allowed: Set[String]): Unit =
    val unknown = options.values.keySet.diff(allowed)
    if unknown.nonEmpty then
      throw IllegalArgumentException(s"Unknown option(s): ${unknown.toSeq.sorted.mkString(", ")}")

  private def requireMaxPositionals(options: CliOptions, max: Int): Unit =
    if options.positionals.length > max then
      throw IllegalArgumentException(s"Expected at most $max positional argument(s), got ${options.positionals.length}")

  private def stringOpt(options: CliOptions, name: String, default: String, positionalIndex: Int): String =
    options.values.getOrElse(name, options.positionals.lift(positionalIndex).getOrElse(default))

  private def intOpt(options: CliOptions, name: String, default: Int, positionalIndex: Int): Int =
    stringOpt(options, name, default.toString, positionalIndex).toInt

  private def longOpt(options: CliOptions, name: String, default: Long, positionalIndex: Int): Long =
    stringOpt(options, name, default.toString, positionalIndex).toLong

  private def doubleOpt(options: CliOptions, name: String, default: Double, positionalIndex: Int): Double =
    stringOpt(options, name, default.toString, positionalIndex).toDouble

  def parseMethod(name: String): HashMethod =
    name.toLowerCase match
      case "biohash"      => HashMethod.BioHash
      case "flyhash"      => HashMethod.FlyHash
      case "naivebiohash" => HashMethod.NaiveBioHash
      case other          => throw IllegalArgumentException(s"Unknown method: $other")

  def parseKs(spec: String): Seq[Int] =
    spec.split(",").map(_.trim.toInt).toSeq

  @main def run(): Unit =
    println(
      "BioHash — subcommands: evalMnist, evalFashion, evalCifar, evalAnn, evalSyntheticText, " +
        "trainTextBenchmark, queryTextBenchmark, microbench, sweepMnist"
    )

  @main def evalMnist(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(options, Set("method", "k", "epochs", "seed", "data-dir", "dataDir"))
    requireMaxPositionals(options, 5)
    val method = stringOpt(options, "method", "biohash", 0)
    val k = intOpt(options, "k", 2, 1)
    val epochs = intOpt(options, "epochs", 5, 2)
    val seed = longOpt(options, "seed", 42L, 3)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .getOrElse(options.positionals.lift(4).getOrElse("data/mnist"))
    val dir = Path.of(dataDir)
    if !Mnist.isAvailable(dir) then
      println(s"MNIST not found in $dir. Run `just download-mnist` to fetch the IDX gzip files.")
      sys.exit(1)
    val split = Mnist.paperSplit(seed)
    val config = EvalConfig(k = k, activity = 0.05, epochs = epochs, seed = seed, method = parseMethod(method))
    val result = EvalRunner.runSplit(split, config, "mnist")
    println(EvalRunner.formatResult(result))

  @main def evalFashion(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(options, Set("method", "k", "epochs", "seed", "data-dir", "dataDir"))
    requireMaxPositionals(options, 5)
    val method = stringOpt(options, "method", "biohash", 0)
    val k = intOpt(options, "k", 2, 1)
    val epochs = intOpt(options, "epochs", 5, 2)
    val seed = longOpt(options, "seed", 42L, 3)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .getOrElse(options.positionals.lift(4).getOrElse("data/fashion-mnist"))
    val dir = Path.of(dataDir)
    if !FashionMnist.isAvailable(dir) then
      println(s"Fashion-MNIST not found in $dir.")
      sys.exit(1)
    val split = FashionMnist.paperSplit(seed)
    val config = EvalConfig(k = k, activity = 0.05, epochs = epochs, seed = seed, method = parseMethod(method))
    val result = EvalRunner.runSplit(split, config, "fashion-mnist")
    println(EvalRunner.formatResult(result))

  @main def evalCifar(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(options, Set("method", "k", "epochs", "seed", "data-dir", "dataDir"))
    requireMaxPositionals(options, 5)
    val method = stringOpt(options, "method", "biohash", 0)
    val k = intOpt(options, "k", 2, 1)
    val epochs = intOpt(options, "epochs", 5, 2)
    val seed = longOpt(options, "seed", 42L, 3)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .getOrElse(options.positionals.lift(4).getOrElse("data/cifar-10-batches-bin"))
    val dir = Path.of(dataDir)
    if !Cifar10.isAvailable(dir) then
      println(s"CIFAR-10 not found in $dir.")
      sys.exit(1)
    val split = Cifar10.paperSplit(seed)
    val config = EvalConfig(k = k, activity = 0.005, epochs = epochs, seed = seed, method = parseMethod(method))
    val result = EvalRunner.runSplit(split, config, "cifar-10")
    println(EvalRunner.formatResult(result))

  @main def evalAnn(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(options, Set("dataset", "k", "epochs", "data-dir", "dataDir"))
    requireMaxPositionals(options, 4)
    val dataset = stringOpt(options, "dataset", "sift10k", 0)
    val k = intOpt(options, "k", 8, 1)
    val epochs = intOpt(options, "epochs", 3, 2)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .getOrElse(options.positionals.lift(3).getOrElse("data/ann"))
    val dir = Path.of(dataDir)
    val ann = dataset.toLowerCase match
      case "sift10k" | "sift1m" =>
        if dataset == "sift10k" then AnnBenchmarks.loadSift10K(dir)
        else AnnBenchmarks.loadSift1M(dir)
      case "gist1m" => AnnBenchmarks.loadGist1M(dir, maxDb = Some(100000), maxQueries = Some(1000))
      case other    => throw IllegalArgumentException(s"Unknown ANN dataset: $other")
    val config = EvalConfig(k = k, activity = 0.05, epochs = epochs)
    val (result, recall) = EvalRunner.runAnnDataset(ann, config)
    println(EvalRunner.formatResult(result.copy(mAP = recall)))

  @main def sweepMnist(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(options, Set("method", "ks", "epochs", "seed", "data-dir", "dataDir"))
    requireMaxPositionals(options, 5)
    val method = stringOpt(options, "method", "biohash", 0)
    val ks = stringOpt(options, "ks", "2,4,8,16,32,64", 1)
    val epochs = intOpt(options, "epochs", 5, 2)
    val seed = longOpt(options, "seed", 42L, 3)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .getOrElse(options.positionals.lift(4).getOrElse("data/mnist"))
    val dir = Path.of(dataDir)
    if !Mnist.isAvailable(dir) then
      println(s"MNIST not found in $dir.")
      sys.exit(1)
    val split = Mnist.paperSplit(seed)
    val base = EvalConfig(activity = 0.05, epochs = epochs, seed = seed, method = parseMethod(method))
    EvalRunner.sweepK(split, parseKs(ks), base, "mnist").foreach(r => println(EvalRunner.formatResult(r)))

  @main def evalSyntheticText(args: String*): Unit =
    val options = parseCli(args)
    requireKnown(
      options,
      Set(
        "method",
        "k",
        "epochs",
        "seed",
        "activity",
        "corpus-size",
        "corpusSize",
        "query-count",
        "queryCount",
        "dim",
        "topics",
        "clusters-per-topic",
        "clustersPerTopic",
        "noise",
        "hard-negative-rate",
        "hardNegativeRate",
        "retrieval-limit",
        "retrievalLimit"
      )
    )
    requireMaxPositionals(options, 0)
    val method = stringOpt(options, "method", "biohash", 0)
    val k = intOpt(options, "k", 8, 0)
    val epochs = intOpt(options, "epochs", 3, 0)
    val seed = longOpt(options, "seed", 42L, 0)
    val activity = doubleOpt(options, "activity", 0.01, 0)
    val corpusSize = options.values
      .get("corpus-size")
      .orElse(options.values.get("corpusSize"))
      .map(_.toInt)
      .getOrElse(50000)
    val queryCount = options.values
      .get("query-count")
      .orElse(options.values.get("queryCount"))
      .map(_.toInt)
      .getOrElse(1000)
    val dim = intOpt(options, "dim", 768, 0)
    val topics = intOpt(options, "topics", 500, 0)
    val clustersPerTopic = options.values
      .get("clusters-per-topic")
      .orElse(options.values.get("clustersPerTopic"))
      .map(_.toInt)
      .getOrElse(4)
    val noise = doubleOpt(options, "noise", 0.15, 0)
    val hardNegativeRate = options.values
      .get("hard-negative-rate")
      .orElse(options.values.get("hardNegativeRate"))
      .map(_.toDouble)
      .getOrElse(0.10)
    val retrievalLimit = options.values
      .get("retrieval-limit")
      .orElse(options.values.get("retrievalLimit"))
      .map(_.toInt)
    val textConfig = TextRetrievalConfig(
      corpusSize = corpusSize,
      queryCount = queryCount,
      dim = dim,
      topics = topics,
      clustersPerTopic = clustersPerTopic,
      noise = noise,
      hardNegativeRate = hardNegativeRate,
      seed = seed
    )
    val split = Synthetic.textRetrievalSplit(textConfig)
    val evalConfig = EvalConfig(
      k = k,
      activity = activity,
      epochs = epochs,
      seed = seed,
      method = parseMethod(method),
      normalizeInputs = true,
      retrievalLimit = retrievalLimit
    )
    val result = EvalRunner.runSplit(split, evalConfig, "synthetic-text")
    val encodeCount = split.databaseVectors.length + split.queryVectors.length
    println(EvalRunner.formatResult(result, Some(encodeCount), Some(split.queryVectors.length)))

  @main def microbench(): Unit =
    println(BenchmarkRunner.format(BenchmarkRunner.runMicrobenchmarks()))

  private def textBenchmarkOptions(options: CliOptions): (String, Path, Path, EvalConfig, Int, Boolean) =
    requireKnown(
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
        "retrieval-limit",
        "retrievalLimit",
        "dense-baseline",
        "denseBaseline"
      )
    )
    requireMaxPositionals(options, 0)
    val dataset = stringOpt(options, "dataset", "scifact", 0)
    val dataDir = options.values
      .get("data-dir")
      .orElse(options.values.get("dataDir"))
      .map(Path.of(_))
      .getOrElse(TextBenchmark.datasetDir(TextBenchmark.DefaultRoot, dataset))
    val artifactRoot = options.values
      .get("artifact-dir")
      .orElse(options.values.get("artifactDir"))
      .map(Path.of(_))
      .getOrElse(Path.of("target", "biohash", "text"))
    val method = stringOpt(options, "method", "biohash", 0)
    val k = intOpt(options, "k", 32, 0)
    val activity = doubleOpt(options, "activity", 0.01, 0)
    val epochs = intOpt(options, "epochs", 3, 0)
    val learningRate = options.values
      .get("learning-rate")
      .orElse(options.values.get("learningRate"))
      .map(_.toDouble)
      .getOrElse(0.01)
    val delta = doubleOpt(options, "delta", 0.0, 0)
    val antiWinnerRank = options.values
      .get("anti-winner-rank")
      .orElse(options.values.get("antiWinnerRank"))
      .map(_.toInt)
      .getOrElse(2)
    val seed = longOpt(options, "seed", 42L, 0)
    val retrievalLimit = options.values
      .get("retrieval-limit")
      .orElse(options.values.get("retrievalLimit"))
      .map(_.toInt)
      .getOrElse(100)
    val denseBaseline = options.values
      .get("dense-baseline")
      .orElse(options.values.get("denseBaseline"))
      .exists(_.equalsIgnoreCase("true"))
    val config = EvalConfig(
      k = k,
      activity = activity,
      epochs = epochs,
      learningRate = learningRate,
      delta = delta,
      antiWinnerRank = antiWinnerRank,
      seed = seed,
      normalizeInputs = true,
      method = parseMethod(method)
    )
    (dataset, dataDir, artifactRoot, config, retrievalLimit, denseBaseline)

  @main def trainTextBenchmark(args: String*): Unit =
    val options = parseCli(args)
    val (dataset, dataDir, artifactRoot, config, _, _) = textBenchmarkOptions(options)
    if !TextBenchmark.isAvailable(dataDir) then
      println(
        s"Text benchmark data not found in $dataDir.\n" +
          "Run `python scripts/prepare_beir_embeddings.py --dataset $dataset` first.\n" +
          "See data/README.md for the expected layout."
      )
      sys.exit(1)
    val loaded = TextBenchmark.load(dataDir, Some(dataset))
    val result = TextBenchmarkRunner.train(loaded, config, artifactRoot)
    println(TextBenchmarkRunner.formatTrainResult(result))

  @main def queryTextBenchmark(args: String*): Unit =
    val options = parseCli(args)
    val (dataset, dataDir, artifactRoot, config, retrievalLimit, denseBaseline) = textBenchmarkOptions(options)
    if !TextBenchmark.isAvailable(dataDir) then
      println(s"Text benchmark data not found in $dataDir.")
      sys.exit(1)
    val artifactDir = TextIndexArtifact.artifactDir(artifactRoot, dataset, config)
    if !Files.exists(artifactDir.resolve(TextIndexArtifact.ManifestFile)) then
      println(
        s"Trained artifact not found in $artifactDir.\n" +
          s"Run `trainTextBenchmark --dataset $dataset` with matching hyperparameters first."
      )
      sys.exit(1)
    val loaded = TextBenchmark.load(dataDir, Some(dataset))
    val result = TextBenchmarkRunner.query(loaded, artifactDir, retrievalLimit, denseBaseline)
    println(TextBenchmarkRunner.formatQueryResult(result))
