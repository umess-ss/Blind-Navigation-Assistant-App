package com.app_for_blind.yolov8tflite

import kotlin.math.max
import kotlin.math.min

class TrackedObject(
    var id: Int,
    var box: BoundingBox,
    var lastSeen: Long,
    var firstSeen: Long = System.currentTimeMillis()
)

class SimpleTracker(
    private val iouThreshold: Float = 0.15f,
    private val maxAgeMs: Long = 500L
) {
    private var nextId = 1
    private var trackedObjects = mutableListOf<TrackedObject>()

    fun update(newBoxes: List<BoundingBox>): List<BoundingBox> {
        val currentTime = System.currentTimeMillis()
        val updatedTrackedObjects = mutableListOf<TrackedObject>()
        val unmatchedNewBoxes = newBoxes.toMutableList()

        // 1. Try to match existing tracked objects with new detections
        for (tracked in trackedObjects) {
            var bestIou = iouThreshold
            var bestMatch: BoundingBox? = null

            for (newBox in unmatchedNewBoxes) {
                val iou = calculateIoU(tracked.box, newBox)
                if (iou > bestIou && tracked.box.clsName == newBox.clsName) {
                    bestIou = iou
                    bestMatch = newBox
                }
            }

            if (bestMatch == null) {
                for (newBox in unmatchedNewBoxes) {
                    if (tracked.box.clsName != newBox.clsName) continue
                    val dx = kotlin.math.abs(tracked.box.cx - newBox.cx)
                    val dy = kotlin.math.abs(tracked.box.cy - newBox.cy)
                    if (dx < 0.20f && dy < 0.20f) {
                        bestMatch = newBox
                        break
                    }
                }
            }

            if (bestMatch != null) {
                // Smooth bounding box movement using EMA (Exponential Moving Average)
                val alpha = 0.7f // More responsive
                val smoothedBox = bestMatch.copy(
                    x1 = tracked.box.x1 * (1 - alpha) + bestMatch.x1 * alpha,
                    y1 = tracked.box.y1 * (1 - alpha) + bestMatch.y1 * alpha,
                    x2 = tracked.box.x2 * (1 - alpha) + bestMatch.x2 * alpha,
                    y2 = tracked.box.y2 * (1 - alpha) + bestMatch.y2 * alpha,
                    cx = tracked.box.cx * (1 - alpha) + bestMatch.cx * alpha,
                    cy = tracked.box.cy * (1 - alpha) + bestMatch.cy * alpha,
                    w = tracked.box.w * (1 - alpha) + bestMatch.w * alpha,
                    h = tracked.box.h * (1 - alpha) + bestMatch.h * alpha
                )
                tracked.box = smoothedBox
                tracked.lastSeen = currentTime
                updatedTrackedObjects.add(tracked)
                unmatchedNewBoxes.remove(bestMatch)
            } else if (currentTime - tracked.lastSeen < maxAgeMs) {
                // Keep track of objects even if not seen in current frame (for a short time)
                updatedTrackedObjects.add(tracked)
            }
        }

        // 2. Create new tracks for unmatched detections
        for (newBox in unmatchedNewBoxes) {
            val newTrack = TrackedObject(
                id = nextId++,
                box = newBox,
                lastSeen = currentTime
            )
            updatedTrackedObjects.add(newTrack)
        }

        trackedObjects = updatedTrackedObjects

        // Return the boxes with assigned IDs
        return trackedObjects.map { tracked ->
            tracked.box.apply {
                this.trackingId = tracked.id
            }
        }
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = max(box1.x1, box2.x1)
        val y1 = max(box1.y1, box2.y1)
        val x2 = min(box1.x2, box2.x2)
        val y2 = min(box1.y2, box2.y2)
        val intersectionArea = max(0F, x2 - x1) * max(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }
}
