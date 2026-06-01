package biohash.data

import java.nio.file.{Files, Path}

object Mnist:

  val DefaultDir = Path.of("data", "mnist")

  def expectedFiles(dir: Path = DefaultDir): Seq[Path] = Seq(
    dir.resolve("train-images-idx3-ubyte.gz"),
    dir.resolve("train-labels-idx1-ubyte.gz"),
    dir.resolve("t10k-images-idx3-ubyte.gz"),
    dir.resolve("t10k-labels-idx1-ubyte.gz")
  )

  def load(dir: Path = DefaultDir): LabeledDataset =
    val trainImages = IdxReader.readImages(dir.resolve("train-images-idx3-ubyte.gz"))
    val trainLabels = IdxReader.readLabels(dir.resolve("train-labels-idx1-ubyte.gz"))
    val testImages = IdxReader.readImages(dir.resolve("t10k-images-idx3-ubyte.gz"))
    val testLabels = IdxReader.readLabels(dir.resolve("t10k-labels-idx1-ubyte.gz"))

    val vectors = trainImages ++ testImages
    val labels = trainLabels ++ testLabels
    LabeledDataset(vectors, labels, "mnist")

  def paperSplit(seed: Long = 42L): RetrievalSplit =
    DatasetSplit.byClass(load(), queriesPerClass = 100, seed = seed)

  def isAvailable(dir: Path = DefaultDir): Boolean =
    expectedFiles(dir).forall(p => Files.exists(p))
