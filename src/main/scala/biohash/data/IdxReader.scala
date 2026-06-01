package biohash.data

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

object IdxReader:

  private val ImageMagic = 2051
  private val LabelMagic = 2049

  def readImages(path: Path): IndexedSeq[Array[Double]] =
    val stream = openStream(path)
    try
      val magic = stream.readInt()
      require(magic == ImageMagic, s"Invalid image magic number: $magic")
      val count = stream.readInt()
      val rows = stream.readInt()
      val cols = stream.readInt()
      val dim = rows * cols
      (0 until count).map { _ =>
        val pixels = new Array[Double](dim)
        var i = 0
        while i < dim do
          pixels(i) = stream.readUnsignedByte().toDouble / 255.0
          i += 1
        pixels
      }.toIndexedSeq
    finally stream.close()

  def readLabels(path: Path): IndexedSeq[Int] =
    val stream = openStream(path)
    try
      val magic = stream.readInt()
      require(magic == LabelMagic, s"Invalid label magic number: $magic")
      val count = stream.readInt()
      (0 until count).map(_ => stream.readUnsignedByte()).toIndexedSeq
    finally stream.close()

  def loadDataset(imagePath: Path, labelPath: Path, name: String): LabeledDataset =
    val vectors = readImages(imagePath)
    val labels = readLabels(labelPath)
    require(vectors.length == labels.length, s"$name: image/label count mismatch")
    LabeledDataset(vectors, labels, name)

  private def openStream(path: Path): DataInputStream =
    val raw = Files.newInputStream(path)
    val buffered = new BufferedInputStream(raw)
    val input =
      if path.getFileName.toString.endsWith(".gz") then
        new DataInputStream(new GZIPInputStream(buffered))
      else new DataInputStream(buffered)
    input
