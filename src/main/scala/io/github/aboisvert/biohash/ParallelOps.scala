// Copyright (c) Alex Boisvert, 2026
// SPDX-License-Identifier: Apache-2.0

package io.github.aboisvert.biohash

import java.util.concurrent.{Callable, Executors, Future}

object ParallelOps:

  private val executor = Executors.newWorkStealingPool()

  def parMap[A, B](items: IndexedSeq[A], threshold: Int)(f: A => B): IndexedSeq[B] =
    if items.length < threshold then items.map(f)
    else
      val futures: IndexedSeq[Future[B]] = items.map { item =>
        executor.submit(new Callable[B]:
          def call(): B = f(item)
        )
      }
      futures.map(_.get()).toIndexedSeq
