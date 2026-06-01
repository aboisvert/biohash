package biohash.eval

/** Information-retrieval metrics for qrels-based text benchmarks. */
object TextRetrievalMetrics:

  final case class MetricSet(
      ndcgAt10: Double,
      mapAt100: Double,
      recallAt10: Double,
      recallAt100: Double
  )

  def evaluate(
      retrievedDocIds: IndexedSeq[IndexedSeq[String]],
      queryIds: IndexedSeq[String],
      qrels: Map[String, Map[String, Int]]
  ): MetricSet =
    require(retrievedDocIds.length == queryIds.length)
    val evaluated = queryIds.flatMap(qrels.get).zip(retrievedDocIds)
    if evaluated.isEmpty then MetricSet(0.0, 0.0, 0.0, 0.0)
    else
      val ndcg = evaluated.map { case (relevant, retrieved) => ndcgAt(retrieved, relevant, 10) }.sum / evaluated.length
      val map = evaluated.map { case (relevant, retrieved) => averagePrecisionAt(retrieved, relevant, 100) }.sum / evaluated.length
      val recall10 = macroRecallAt(evaluated, 10)
      val recall100 = macroRecallAt(evaluated, 100)
      MetricSet(ndcg, map, recall10, recall100)

  def ndcgAt(retrievedDocIds: IndexedSeq[String], relevant: Map[String, Int], k: Int): Double =
    val cutoff = math.min(k, retrievedDocIds.length)
    if relevant.isEmpty then 0.0
    else
      val dcg = (0 until cutoff).map { rank =>
        val rel = relevant.getOrElse(retrievedDocIds(rank), 0).toDouble
        if rel <= 0.0 then 0.0 else rel / (math.log(rank + 2.0) / math.log(2.0))
      }.sum
      val ideal = relevant.values.toSeq.sortBy(-_).take(k)
      val idcg = ideal.zipWithIndex.map { case (rel, rank) =>
        rel.toDouble / (math.log(rank + 2.0) / math.log(2.0))
      }.sum
      if idcg <= 0.0 then 0.0 else dcg / idcg

  def averagePrecisionAt(retrievedDocIds: IndexedSeq[String], relevant: Map[String, Int], k: Int): Double =
    val cutoff = math.min(k, retrievedDocIds.length)
    val totalRelevant = relevant.count(_._2 > 0)
    if totalRelevant == 0 then 0.0
    else
      var hits = 0
      var sumPrecision = 0.0
      var rank = 0
      while rank < cutoff do
        if relevant.getOrElse(retrievedDocIds(rank), 0) > 0 then
          hits += 1
          sumPrecision += hits.toDouble / (rank + 1)
        rank += 1
      sumPrecision / totalRelevant

  /** Macro-averaged recall: sum of found relevant docs / sum of all relevant docs. */
  def macroRecallAt(
      evaluated: IndexedSeq[(Map[String, Int], IndexedSeq[String])],
      k: Int
  ): Double =
    val totals = evaluated.map { case (relevant, retrieved) =>
      val relevantIds = relevant.filter(_._2 > 0).keySet
      if relevantIds.isEmpty then (0, 0)
      else
        val found = retrieved.take(k).count(relevantIds.contains)
        (found, relevantIds.size)
    }
    val foundTotal = totals.map(_._1).sum
    val relevantTotal = totals.map(_._2).sum
    if relevantTotal == 0 then 0.0 else foundTotal.toDouble / relevantTotal

  def formatMetrics(metrics: MetricSet, prefix: String = ""): String =
    f"${prefix}nDCG@10=${metrics.ndcgAt10}%.4f MAP@100=${metrics.mapAt100}%.4f " +
      f"Recall@10=${metrics.recallAt10}%.4f Recall@100=${metrics.recallAt100}%.4f"
