// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package biohash.data

/** A labeled dataset of feature vectors. */
final case class LabeledDataset(
    vectors: IndexedSeq[Array[Double]],
    labels: IndexedSeq[Int],
    name: String
):

  def size: Int = vectors.length

  def inputDim: Int = if vectors.isEmpty then 0 else vectors.head.length

/** Paper-compatible train/query/database split. */
final case class RetrievalSplit(
    trainVectors: IndexedSeq[Array[Double]],
    databaseVectors: IndexedSeq[Array[Double]],
    databaseLabels: IndexedSeq[Int],
    queryVectors: IndexedSeq[Array[Double]],
    queryLabels: IndexedSeq[Int]
)

object DatasetSplit:

  /** MNIST/CIFAR protocol: sample queriesPerClass from each class; rest is train+database. */
  def byClass(
      dataset: LabeledDataset,
      queriesPerClass: Int,
      seed: Long = 42L
  ): RetrievalSplit =
    import scala.util.Random
    val rng = Random(seed)

    val byLabel = dataset.labels.zipWithIndex.groupBy(_._1).view.mapValues(_.map(_._2).toList).toMap
    val numClasses = byLabel.keys.max + 1

    val queryIndices = scala.collection.mutable.ArrayBuffer.empty[Int]
    val poolIndices = scala.collection.mutable.ArrayBuffer.empty[Int]

    (0 until numClasses).foreach { c =>
      val indices = rng.shuffle(byLabel.getOrElse(c, Nil))
      val (queries, rest) = indices.splitAt(math.min(queriesPerClass, indices.length))
      queryIndices ++= queries
      poolIndices ++= rest
    }

    RetrievalSplit(
      trainVectors = poolIndices.map(i => dataset.vectors(i)).toIndexedSeq,
      databaseVectors = poolIndices.map(i => dataset.vectors(i)).toIndexedSeq,
      databaseLabels = poolIndices.map(i => dataset.labels(i)).toIndexedSeq,
      queryVectors = queryIndices.map(i => dataset.vectors(i)).toIndexedSeq,
      queryLabels = queryIndices.map(i => dataset.labels(i)).toIndexedSeq
    )
