package com.app_for_blind.yolov8tflite

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String,
    var distance: String = "", 
    var isAnnounced: Boolean = false, 
    var trackingId: Int = -1, 
    var depthCategory: String = "FAR",
    var rawDepthScore: Float = 0f, // Normalized inverse depth (0..1, 1 is close)
    var distanceInMeters: Float = 0f // Estimated metric distance
)
