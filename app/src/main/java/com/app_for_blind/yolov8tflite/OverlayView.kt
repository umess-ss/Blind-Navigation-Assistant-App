package com.app_for_blind.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var announcedBoxPaint = Paint() 
    private var lockedBoxPaint = Paint() 
    private var lostBoxPaint = Paint() // Red for lost
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()
    
    private var lockedObject: String? = null
    private var currentState: NavigationSystem.NavState = NavigationSystem.NavState.IDLE

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        announcedBoxPaint.reset()
        lockedBoxPaint.reset()
        lostBoxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = Color.BLUE
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
        
        announcedBoxPaint.color = Color.CYAN
        announcedBoxPaint.strokeWidth = 8F
        announcedBoxPaint.style = Paint.Style.STROKE
        
        lockedBoxPaint.color = Color.GREEN
        lockedBoxPaint.strokeWidth = 10F 
        lockedBoxPaint.style = Paint.Style.STROKE

        lostBoxPaint.color = Color.RED
        lostBoxPaint.strokeWidth = 10F
        lostBoxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            val isLocked = lockedObject != null && it.clsName.equals(lockedObject, ignoreCase = true)
            
            // Choose paint based on status and state
            val currentPaint = when {
                isLocked && (currentState == NavigationSystem.NavState.LOST || currentState == NavigationSystem.NavState.TARGET_REACHED) -> lostBoxPaint
                isLocked -> lockedBoxPaint
                it.isAnnounced -> announcedBoxPaint
                else -> boxPaint
            }
            
            canvas.drawRect(left, top, right, bottom, currentPaint)
            
            val drawableText = String.format(java.util.Locale.US, "%s [ID:%d] %.1fm", it.clsName, it.trackingId, it.distanceInMeters)

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>, state: NavigationSystem.NavState) {
        results = boundingBoxes
        currentState = state
        invalidate()
    }
    
    fun setLockedObject(objectName: String?) {
        lockedObject = objectName
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
