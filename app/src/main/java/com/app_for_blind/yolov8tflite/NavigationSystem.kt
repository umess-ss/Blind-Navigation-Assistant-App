package com.app_for_blind.yolov8tflite

import android.util.Log
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

class NavigationSystem(private val tts: (String, String) -> Unit) {

    enum class NavState {
        IDLE,
        SCANNING,
        WAITING_FOR_COMMAND,
        OBJECT_SELECTED,
        TRACKING,
        APPROACHING,
        VERY_CLOSE,
        TARGET_REACHED,
        LOST,
        REFOUND,
        OBSTACLE_ALERT
    }

    private var currentState = NavState.IDLE
    private var targetId: Int = -1
    private var targetName: String? = null
    
    private var lastGuidanceTime = 0L
    private val GUIDANCE_INTERVAL_MS = 3000L 
    private val VERY_CLOSE_INTERVAL_MS = 1500L 

    private var lastAccelTime = 0L
    private val ACCEL_COOLDOWN_MS = 4000L 

    private var lastObstacleTime = 0L
    private val OBSTACLE_COOLDOWN_MS = 2500L 

    // Lost-state navigation memory
    private var lastKnownCx: Float = 0.5f
    private var lastKnownCy: Float = 0.5f
    private var lastKnownDist: Float = 5.0f
    private var lostStartTime: Long = 0L
    private var lastLostGuidanceTime: Long = 0L
    private val LOST_GUIDANCE_INTERVAL_MS = 3000L
    
    // Depth Baseline Lock variables
    private var selectionDepth: Float = -1f
    private var selectionBBoxArea: Float = -1f
    private var smoothedDist: Float = -1f
    private var pendingBaselineCapture = false
    
    // Target Reached stability
    private var reachedConfirmationStartTime = 0L
    private var lastFarTime = 0L
    
    private var lostFrames = 0
    private val LOST_THRESHOLD_FRAMES = 20
    
    private var lostVoiceFired = false
    private var refoundVoiceFired = false
    private var lastObjectDetectedTime = 0L

    // Direction and Depth constants
    private val DEAD_ZONE_X = 0.12f 

    fun reset() {
        currentState = NavState.IDLE
        targetId = -1
        targetName = null
        lastGuidanceTime = 0
        lastAccelTime = 0
        lastObstacleTime = 0
        lastKnownCx = 0.5f
        lastKnownCy = 0.5f
        lastKnownDist = 5.0f
        lostStartTime = 0L
        lastLostGuidanceTime = 0L
        selectionDepth = -1f
        selectionBBoxArea = -1f
        smoothedDist = -1f
        pendingBaselineCapture = false
        reachedConfirmationStartTime = 0L
        lastFarTime = 0L
        lostFrames = 0
        lostVoiceFired = false
        refoundVoiceFired = false
        lastObjectDetectedTime = 0L
    }

    fun startScanning() {
        transitionTo(NavState.SCANNING)
    }

    fun setWaitingState() {
        transitionTo(NavState.WAITING_FOR_COMMAND)
    }

    fun setTarget(id: Int, name: String) {
        targetId = id
        targetName = name
        transitionTo(NavState.OBJECT_SELECTED)
        pendingBaselineCapture = true
        reachedConfirmationStartTime = 0L
        lostFrames = 0
        lostVoiceFired = false
        refoundVoiceFired = false
    }

    private fun transitionTo(newState: NavState) {
        if (currentState != newState) {
            Log.d("NavSystem", "Transition: $currentState -> $newState")
            
            when (newState) {
                NavState.TARGET_REACHED -> { /* MainActivity handles this announcement */ }
                NavState.LOST -> {
                    // One-shot voice triggered in update() for more control
                }
                NavState.REFOUND -> {
                    // One-shot voice triggered in update() for more control
                }
                NavState.OBJECT_SELECTED -> tts("Object selected. Starting tracking.", "LOW_PRIORITY")
                NavState.SCANNING -> tts("Scanning for objects.", "LOW_PRIORITY")
                else -> {}
            }
            currentState = newState
        }
    }

    fun update(boxes: List<BoundingBox>, accelX: Float, accelY: Float, accelZ: Float) {
        val currentTime = System.currentTimeMillis()

        if (currentState == NavState.IDLE || currentState == NavState.TARGET_REACHED || currentState == NavState.WAITING_FOR_COMMAND) {
            return
        }

        // --- TARGET DETECTION & PERSISTENCE (FIXED) ---
        var target = boxes.find { it.trackingId == targetId }
        if (target == null && targetName != null) {
            // ID drifted — find closest box with same class name
            target = boxes.filter {
                it.clsName.equals(targetName, ignoreCase = true)
            }.minByOrNull { it.distanceInMeters }
            // Re-lock onto new ID so future frames match directly
            if (target != null) {
                targetId = target.trackingId
            }
        }
        val objectDetected = target != null

        if (objectDetected) {
            val wasLost = currentState == NavState.LOST
            if (lostFrames == 0 && objectDetected && target != null) {
                lastKnownCx = target.cx
                lastKnownCy = target.cy
                lastKnownDist = target.distanceInMeters
            }
            lostFrames = 0
            lastObjectDetectedTime = currentTime
            lostVoiceFired = false

            if (wasLost && !refoundVoiceFired) {
                refoundVoiceFired = true
                val dist = String.format(Locale.US, "%.1f meters", target?.distanceInMeters ?: 0f)
                val dir = getDirectionText(target?.cx ?: 0.5f)
                tts("Target found again. $dist $dir.", "HIGH_PRIORITY")
                currentState = NavState.TRACKING
                lastGuidanceTime = currentTime
            }
        } else {
            refoundVoiceFired = false
            lostFrames++
        }

        // --- PRIORITY 1: TARGET LOST ---
        if (lostFrames >= LOST_THRESHOLD_FRAMES) {
            if (currentState != NavState.LOST) {
                currentState = NavState.LOST
                if (!lostVoiceFired) {
                    lostVoiceFired = true
                    lostStartTime = currentTime
                    lastLostGuidanceTime = currentTime
                    val dir = getDirectionText(lastKnownCx)
                    val dist = String.format(Locale.US, "%.1f", lastKnownDist)
                    tts("Target lost. The ${targetName ?: "object"} was last seen $dir, $dist meters away.", "HIGH_PRIORITY")
                }
            }
            
            // Periodic lost guidance while still searching
            if (currentTime - lastLostGuidanceTime > LOST_GUIDANCE_INTERVAL_MS) {
                lastLostGuidanceTime = currentTime
                val lostDuration = currentTime - lostStartTime
                val searchInstruction = buildLostSearchInstruction(lostDuration)
                tts(searchInstruction, "HIGH_PRIORITY")
            }

            // Accelerometer assistance ONLY in LOST state
            if (currentTime - lastAccelTime > ACCEL_COOLDOWN_MS) {
                if (accelX > 3.0f) {
                    tts("Tilt your phone slightly left.", "ACCEL")
                    lastAccelTime = currentTime
                } else if (accelX < -3.0f) {
                    tts("Tilt your phone slightly right.", "ACCEL")
                    lastAccelTime = currentTime
                } else if (accelY > 4.0f) {
                    tts("Raise your phone slightly.", "ACCEL")
                    lastAccelTime = currentTime
                } else if (accelY < -4.0f) {
                    tts("Lower your phone slightly.", "ACCEL")
                    lastAccelTime = currentTime
                }
            }
            return
        }

        if (!objectDetected) return

        val currentTarget = target!!
        lastKnownCx = currentTarget.cx

        // --- DEPTH & DISTANCE LOGIC (FIXED) ---
        
        // SOURCE OF TRUTH: use distanceInMeters already set by DepthEstimator
        val rawDist = currentTarget.distanceInMeters.coerceIn(0.1f, 8.0f)
        val currentArea = currentTarget.w * currentTarget.h
        
        // Baseline lock: capture at selection moment
        if (pendingBaselineCapture) {
            selectionDepth = rawDist
            selectionBBoxArea = currentArea.coerceAtLeast(0.001f)
            smoothedDist = selectionDepth
            pendingBaselineCapture = false
        }
        
        // Hybrid estimate: bbox growth tells us how much closer we are
        val safeArea = currentArea.coerceAtLeast(0.001f)
        val bboxRatio = safeArea / selectionBBoxArea.coerceAtLeast(0.001f)
        val bboxBasedDist = (selectionDepth / sqrt(bboxRatio)).coerceIn(0.1f, 8.0f)
        
        // EMA smoothing
        val newEstimate = if (abs(1.0f - bboxRatio) < 0.08f) {
            rawDist
        } else {
            bboxBasedDist
        }
        
        smoothedDist = (0.75f * smoothedDist + 0.25f * newEstimate).coerceIn(0.1f, 8.0f)
        
        // Always take minimum of raw and smoothed (safety)
        val finalDist = min(smoothedDist, rawDist).coerceIn(0.1f, 8.0f)
        currentTarget.distanceInMeters = finalDist
        
        val dx = currentTarget.cx - 0.5f 
        val isCentered = abs(dx) <= DEAD_ZONE_X

        // --- PRIORITY 3: TARGET REACHED ---
        val isReachedCandidate = finalDist <= 1.1f && currentArea >= 0.08f && currentTarget.cnf >= 0.60f
        val wasFarRecently = (currentTime - lastFarTime) < 3000L

        if (isReachedCandidate && isCentered && !wasFarRecently) {
            if (reachedConfirmationStartTime == 0L) {
                reachedConfirmationStartTime = currentTime
            } else if (currentTime - reachedConfirmationStartTime >= 500L) {
                transitionTo(NavState.TARGET_REACHED)
                return 
            }
        } else {
            reachedConfirmationStartTime = 0L
        }
        
        if (finalDist > 2.5f) {
            lastFarTime = currentTime
        }

        // --- PRIORITY 4: OBSTACLE ALERT ---
        val obstacle = boxes.find { 
            it.trackingId != targetId && 
            it.rawDepthScore > 0.8f && 
            it.cx in 0.3..0.7
        }

        if (obstacle != null) {
            if (currentTime - lastObstacleTime > OBSTACLE_COOLDOWN_MS) {
                transitionTo(NavState.OBSTACLE_ALERT)
                val turnDir = if (accelX > 0) "left" else "right"
                tts("Obstacle ahead. Turn slightly $turnDir.", "HIGH_PRIORITY")
                lastObstacleTime = currentTime
            }
            return
        } else if (currentState == NavState.OBSTACLE_ALERT) {
            transitionTo(NavState.TRACKING)
        }

        // --- PRIORITY 5: CONTINUOUS GUIDANCE ---
        val guidanceInterval = if (finalDist < 2.0f) VERY_CLOSE_INTERVAL_MS else GUIDANCE_INTERVAL_MS
        
        if (currentTime - lastGuidanceTime > guidanceInterval) {
            
            val directionInstruction = when {
                currentTarget.cx < 0.38f -> "Turn left."
                currentTarget.cx > 0.62f -> "Turn right."
                else -> "Go straight."
            }
            
            // Depth Saturation Fix
            val isSaturated = finalDist <= 1.5f && currentArea < 0.15f
            
            val distText = String.format(Locale.US, "%.1f meters", finalDist)
            val voiceOutput = when {
                isSaturated -> "Target is far ahead. Continue forward."
                finalDist <= 2.0f -> "Very close, $distText $directionInstruction"
                else -> "$directionInstruction, $distText."
            }

            // Sync states
            if (finalDist <= 2.0f && finalDist > 1.1f) {
                transitionTo(NavState.VERY_CLOSE)
            } else if (finalDist > 2.0f) {
                transitionTo(NavState.TRACKING)
            }

            val voiceId = if (finalDist <= 2.0f) "NAV_CLOSE" else "NAV_GUIDANCE"
            tts(voiceOutput, voiceId)
            lastGuidanceTime = currentTime
        }
    }

    private fun buildLostSearchInstruction(lostDurationMs: Long): String {
        val name = targetName ?: "object"
        val dir = getDirectionText(lastKnownCx)
        val dist = String.format(Locale.US, "%.1f", lastKnownDist)

        return when {
            lostDurationMs < 2000L -> {
                // Phase 1: Object just slipped — small scan
                "Slow down. Scan $dir to find the $name."
            }
            lostDurationMs < 5000L -> {
                // Phase 2: May have moved past it
                when {
                    lastKnownCx < 0.38f -> "Turn left and scan slowly. The $name was on your left."
                    lastKnownCx > 0.62f -> "Turn right and scan slowly. The $name was on your right."
                    else -> "Stop and scan ahead. The $name was $dist meters in front of you."
                }
            }
            else -> {
                // Phase 3: Strong recovery — turn around if needed
                when {
                    lastKnownCx < 0.38f -> "Turn left. The $name was last seen $dir, $dist meters away."
                    lastKnownCx > 0.62f -> "Turn right. The $name was last seen $dir, $dist meters away."
                    else -> "Turn around slowly. The $name was last seen straight ahead, $dist meters away."
                }
            }
        }
    }

    private fun getDirectionText(cx: Float): String = when {
        cx < 0.38f -> "to your left"
        cx > 0.62f -> "to your right"
        else -> "straight ahead"
    }

    fun getCurrentState(): NavState = currentState
}
