package biohash.data

import java.nio.file.{Files, Path}

object FashionMnist:

  val DefaultDir = Path.of("data", "fashion-mnist")

  def load(dir: Path = DefaultDir): LabeledDataset =
    val trainImages = IdxReader.readImages(dir.resolve("train-images-idx3-ubyte.gz"))
    val trainLabels = IdxReader.readLabels(dir.resolve("train-labels-idx1-ubyte.gz"))
    val testImages = IdxReader.readImages(dir.resolve("t10k-images-idx3-ubyte.gz"))
    val testLabels = IdxReader.readLabels(dir.resolve("t10k-labels-idx1-ubyte.gz"))
    LabeledDataset(trainImages ++ testImages, trainLabels ++ testLabels, "fashion-mnist")

  def paperSplit(seed: Long = 42L): RetrievalSplit =
    DatasetSplit.byClass(load(), queriesPerClass = 100, seed = seed)

  def isAvailable(dir: Path = DefaultDir): Boolean =
    Seq(
      "train-images-idx3-ubyte.gz",
      "train-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz",
      "t10k-labels-idx1-ubyte.gz"
    ).map(name => dir.resolve(name)).forall(p => Files.exists(p))
