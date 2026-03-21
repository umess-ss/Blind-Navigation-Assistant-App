package com.app_for_blind.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

// DepthEstimator class handles MiDaS TFLite model for depth estimation
class DepthEstimator(context: Context) {
    private lateinit var interpreter: Interpreter
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0

    init {
        val compatList = CompatibilityList()
        val options = Interpreter.Options()
        
        // Try initializing with GPU
        var gpuDelegate: GpuDelegate? = null
        var initialized = false
        
        if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                
                val modelFile = FileUtil.loadMappedFile(context, Constants.DEPTH_MODEL_PATH)
                interpreter = Interpreter(modelFile, options)
                initialized = true
            } catch (e: Exception) {
                Log.e("DepthEstimator", "GPU initialization failed, falling back to CPU", e)
                // Clean up failed delegate
                if (gpuDelegate != null) {
                    gpuDelegate.close()
                    gpuDelegate = null
                }
                options.delegates.clear()
            }
        }
        
        if (!initialized) {
            // Fallback to CPU
            val cpuOptions = Interpreter.Options().apply { setNumThreads(4) }
            val modelFile = FileUtil.loadMappedFile(context, Constants.DEPTH_MODEL_PATH)
            interpreter = Interpreter(modelFile, cpuOptions)
        }

        val inputShape = interpreter.getInputTensor(0).shape() 
        // Handle shapes like [1, 256, 256, 3] (NHWC) or [1, 3, 256, 256] (NCHW)
        if (inputShape[1] == 3) {
            inputImageHeight = inputShape[2]
            inputImageWidth = inputShape[3]
        } else {
            inputImageHeight = inputShape[1]
            inputImageWidth = inputShape[2]
        }
    }

    // Process the frame and return a depth map (float array)
    fun computeDepthMap(bitmap: Bitmap): FloatArray {
        // 1. Resize and Preprocess Input
        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0f, 255f)) // Normalize [0, 255] -> [0, 1]
            .build()
            
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // 2. Run Inference
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        
        interpreter.run(processedImage.buffer, outputBuffer.buffer)
        
        return outputBuffer.floatArray
    }
    
    /**
     * Estimates relative depth category for a given bounding box.
     * Uses normalization per frame and samples the center 40% of the bounding box.
     */
    fun processDepthForBoxes(depthMap: FloatArray, boxes: List<BoundingBox>) {
        if (depthMap.isEmpty()) return

        // 1. Normalize depth map to 0..1 range (MiDaS is inverse relative)
        val minVal = depthMap.minOrNull() ?: 0f
        val maxVal = depthMap.maxOrNull() ?: 1f
        val range = if (maxVal > minVal) maxVal - minVal else 1f

        boxes.forEach { box ->
            // 2. Sample center 40% of bounding box
            val sampleBoxX1 = (box.x1 + box.w * 0.3f) * inputImageWidth
            val sampleBoxY1 = (box.y1 + box.h * 0.3f) * inputImageHeight
            val sampleBoxX2 = (box.x1 + box.w * 0.7f) * inputImageWidth
            val sampleBoxY2 = (box.y1 + box.h * 0.7f) * inputImageHeight

            var sumNormalizedDepth = 0f
            var count = 0

            val startX = sampleBoxX1.toInt().coerceIn(0, inputImageWidth - 1)
            val endX = sampleBoxX2.toInt().coerceIn(0, inputImageWidth - 1)
            val startY = sampleBoxY1.toInt().coerceIn(0, inputImageHeight - 1)
            val endY = sampleBoxY2.toInt().coerceIn(0, inputImageHeight - 1)

            for (y in startY..endY) {
                for (x in startX..endX) {
                    val index = y * inputImageWidth + x
                    if (index < depthMap.size) {
                        val rawVal = depthMap[index]
                        val normalized = (rawVal - minVal) / range
                        sumNormalizedDepth += normalized
                        count++
                    }
                }
            }

            if (count > 0) {
                val avgDepth = sumNormalizedDepth / count
                box.rawDepthScore = avgDepth
                box.depthCategory = categorizeDepth(avgDepth)
                
                // MiDaS: avgDepth is normalized 0..1, higher = closer (inverse depth)
                // Empirically tuned: 0.9 avgDepth ≈ 0.3m, 0.1 avgDepth ≈ 5m
                box.distanceInMeters = (0.8f / (avgDepth + 0.12f)).coerceIn(0.2f, 8.0f)
                box.distance = String.format("%.1f meters", box.distanceInMeters)
            } else {
                box.rawDepthScore = 0f
                box.depthCategory = "FAR"
                box.distanceInMeters = 5.0f
                box.distance = "FAR"
            }
        }
    }

    private fun categorizeDepth(normalizedDepth: Float): String {
        // MiDaS output is inverse depth: larger value = closer
        return when {
            normalizedDepth > 0.8f -> "VERY_CLOSE"
            normalizedDepth > 0.6f -> "CLOSE"
            normalizedDepth > 0.3f -> "MEDIUM"
            else -> "FAR"
        }
    }

    fun close() {
        interpreter.close()
    }
}
