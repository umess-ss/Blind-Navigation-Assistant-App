package com.app_for_blind.yolov8tflite

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.app_for_blind.yolov8tflite.Constants.LABELS_PATH
import com.app_for_blind.yolov8tflite.Constants.MODEL_PATH
import yolov8tflite.R
import yolov8tflite.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity(), Detector.DetectorListener, TextToSpeech.OnInitListener, SensorEventListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var tracker = SimpleTracker()
    private lateinit var navigationSystem: NavigationSystem

    private lateinit var cameraExecutor: ExecutorService
    private var tts: TextToSpeech? = null
    
    @Volatile
    private var isSpeaking = false
    private var lastSpeakTime = 0L
    
    // Voice Selection & Recognition
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var recognitionIntent: Intent
    private var canStartListening = true
    private var framesSinceError = 0
    private val REQUIRED_FRAMES_AFTER_ERROR = 30 

    // Accelerometer Sensors
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var accelX: Float = 0f
    private var accelY: Float = 0f
    private var accelZ: Float = 0f

    // Startup Logic
    private var isStartupPhase = true
    private val startupDetections = mutableSetOf<String>()
    
    // Gesture Detector
    private lateinit var gestureDetector: GestureDetector
    
    // Application States
    enum class AppState {
        STARTUP,
        DETECTING,
        LISTENING,
        NAVIGATING,
        COMPLETED,
        SCENE_DESCRIPTION 
    }
    
    private var currentState = AppState.STARTUP
    private var targetObjectName: String? = null
    
    // Performance
    private var frameCount = 0
    private val MIDAS_SKIP_FRAMES = 5
    @Volatile
    private var currentDepthMap: FloatArray? = null
    private var lastTrackedBoxes: List<BoundingBox> = emptyList()

    // Announcements
    private val lastSpokenTimes = ConcurrentHashMap<String, Long>()
    private val SCAN_ANNOUNCEMENT_COOLDOWN = 20000L
    private val announcedObjects = ConcurrentHashMap<String, Boolean>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tts = TextToSpeech(this, this)
        navigationSystem = NavigationSystem { text, id -> speak(text, id) }

        initializeSpeechRecognizer()
        initializeGestureDetector()
        initializeSensors()

        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) { toast(it) }
            try {
                depthEstimator = DepthEstimator(baseContext)
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing DepthEstimator", e)
            }
        }

        if (allPermissionsGranted()) startCamera()
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)

        bindListeners()
        
        navigationSystem.startScanning()
        
        // Handle Startup Object List
        Handler(Looper.getMainLooper()).postDelayed({
            finishStartupPhase()
        }, 3000)
    }

    private fun finishStartupPhase() {
        if (!isStartupPhase) return
        isStartupPhase = false
        
        val objectsList = startupDetections.take(15).joinToString(", ")
        val startupMsg = if (objectsList.isNotEmpty()) {
            "I see: $objectsList. Double tap and say the object you want to find."
        } else {
            "I don't see anything yet. Double tap and say the object you want to find."
        }
        
        speak(startupMsg, "STARTUP_DONE")
        currentState = AppState.DETECTING
        navigationSystem.setWaitingState()
    }

    private fun initializeSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    private fun initializeGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                handleDoubleTap()
                return true
            }
            override fun onFling(e1: MotionEvent?, e2: MotionEvent, vX: Float, vY: Float): Boolean {
                if (e1 == null) return false
                val diffX = e2.x - e1.x
                if (abs(diffX) > 100 && abs(vX) > 100) {
                    if (diffX > 0) handleSwipeRight() else handleSwipeLeft()
                    return true
                }
                return false
            }
        })
    }
    
    private fun handleDoubleTap() {
        if (!canStartListening) return
        tts?.stop()
        isSpeaking = false 
        currentState = AppState.LISTENING
        speak("What would you like to find?", "MANUAL_LISTEN")
    }

    private fun handleSwipeRight() {
        tts?.stop()
        isSpeaking = false
        currentState = AppState.DETECTING
        targetObjectName = null
        navigationSystem.reset()
        binding.overlay.setLockedObject(null)
        announcedObjects.clear()
        lastSpokenTimes.clear()
        binding.bottomNavigation.selectedItemId = R.id.navigation_camera
        speak("System reset. Looking for objects.", "RESET_SWIPE")
    }

    private fun handleSwipeLeft() {
        if (currentState != AppState.SCENE_DESCRIPTION) {
            tts?.stop()
            isSpeaking = false
            currentState = AppState.SCENE_DESCRIPTION
            binding.bottomNavigation.selectedItemId = R.id.navigation_scene
            pendingSceneDescription = true
        }
    }
    
    private var pendingSceneDescription = false

    override fun onTouchEvent(event: MotionEvent?): Boolean = event?.let { gestureDetector.onTouchEvent(it) || super.onTouchEvent(it) } ?: super.onTouchEvent(event)

    private fun initializeSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        recognitionIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }
        
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                if (currentState == AppState.LISTENING) {
                     speak("I didn't catch that. Try again later.", "ERROR_SILENT")
                     currentState = AppState.DETECTING
                     framesSinceError = 0
                     canStartListening = false
                }
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) handleUserSpeech(matches[0])
                else {
                    speak("I didn't hear anything.", "ERROR_SILENT")
                    currentState = AppState.DETECTING
                }
            }
            override fun onPartialResults(pResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }
    
    private fun handleUserSpeech(text: String) {
        val raw = text.lowercase(Locale.ROOT).trim()
        Log.d("Speech", "Input: $raw")
        
        if (raw.contains("reset") || raw.contains("stop")) {
            handleSwipeRight()
            return
        }

        val target = extractObjectName(raw)
        if (target.isEmpty()) {
            speak("Object not recognized.", "ERROR_SILENT")
            currentState = AppState.DETECTING
            return
        }

        val normalizedTarget = normalizeClassName(target)
        
        val match = lastTrackedBoxes.filter { box ->
            val normDetected = normalizeClassName(box.clsName.lowercase(Locale.ROOT))
            normalizedTarget.contains(normDetected) || normDetected.contains(normalizedTarget)
        }.minByOrNull { it.distanceInMeters }
        
        if (match != null) {
            targetObjectName = match.clsName
            currentState = AppState.NAVIGATING
            navigationSystem.setTarget(match.trackingId, match.clsName)
            binding.overlay.setLockedObject(match.clsName)
            speak("Target locked on ${match.clsName}.", "NAV_START")
        } else {
            speak("I don't see a $target right now.", "DESCRIBE")
            currentState = AppState.DETECTING
        }
    }

    private fun normalizeClassName(name: String): String = when {
        name.contains("cup") || name.contains("mug") -> "cup"
        name.contains("bowl") || name.contains("dish") -> "bowl"
        name.contains("cell phone") || name.contains("phone") -> "cell phone"
        name.contains("tv") || name.contains("monitor") -> "tv"
        else -> name
    }

    private fun extractObjectName(text: String): String {
        val prefixes = listOf("find ", "track ", "navigate to ", "go to ", "where is ", "looking for ")
        var result = text
        for (p in prefixes) if (result.startsWith(p)) { result = result.substring(p.length); break }
        return result.trim()
    }

    private fun bindListeners() {
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.navigation_camera -> { if (currentState == AppState.SCENE_DESCRIPTION) handleSwipeRight(); true }
                R.id.navigation_scene -> { if (currentState != AppState.SCENE_DESCRIPTION) handleSwipeLeft(); true }
                else -> false
            }
        }
    }

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            cameraProvider = ProcessCameraProvider.getInstance(this).get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val rotation = binding.viewFinder.display.rotation
        
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).setTargetRotation(rotation).build()
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val buffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            imageProxy.use { buffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            val rotated = Bitmap.createBitmap(buffer, 0, 0, buffer.width, buffer.height, Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }, true)

            if (frameCount % MIDAS_SKIP_FRAMES == 0) currentDepthMap = depthEstimator?.computeDepthMap(rotated)
            frameCount++
            detector?.detect(rotated)
        }

        provider.unbindAll()
        try {
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            preview?.surfaceProvider = binding.viewFinder.surfaceProvider
        } catch(e: Exception) { Log.e(TAG, "Binding failed", e) }
    }

    override fun onEmptyDetect() {
        runOnUiThread { binding.overlay.clear() }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        val tracked = tracker.update(boundingBoxes)
        currentDepthMap?.let { depthEstimator?.processDepthForBoxes(it, tracked) }
        lastTrackedBoxes = tracked

        if (isStartupPhase) {
            boundingBoxes.forEach { startupDetections.add(it.clsName) }
        }

        if (!canStartListening) {
            framesSinceError++
            if (framesSinceError >= REQUIRED_FRAMES_AFTER_ERROR) canStartListening = true
        }

        runOnUiThread {
            binding.overlay.setResults(tracked, navigationSystem.getCurrentState())
            binding.overlay.invalidate()
            val now = System.currentTimeMillis()

            if (currentState == AppState.SCENE_DESCRIPTION && pendingSceneDescription) {
                pendingSceneDescription = false
                generateSceneDescription(tracked)
            } else {
                when (currentState) {
                    AppState.DETECTING -> handleDetectingState(tracked, now)
                    AppState.NAVIGATING -> {
                        navigationSystem.update(tracked, accelX, accelY, accelZ)
                        if (navigationSystem.getCurrentState() == NavigationSystem.NavState.TARGET_REACHED) {
                            currentState = AppState.COMPLETED
                            speak("Target reached. You have arrived at the ${targetObjectName ?: "object"}.", "REACHED_FINAL")
                            Handler(Looper.getMainLooper()).postDelayed({
                                handleSwipeRight()
                            }, 5000)
                        }
                    }
                    AppState.COMPLETED -> {}
                    else -> {}
                }
            }
        }
    }
    
    private fun handleDetectingState(boxes: List<BoundingBox>, now: Long) {
        if (isSpeaking) return
        boxes.forEach { box ->
            if (announcedObjects[box.clsName] != true) {
                val last = lastSpokenTimes[box.clsName] ?: 0L
                if (now - last > SCAN_ANNOUNCEMENT_COOLDOWN) {
                    announcedObjects[box.clsName] = true
                    lastSpokenTimes[box.clsName] = now
                    val dist = String.format(java.util.Locale.US, "%.1f meters", box.distanceInMeters)
                    speak("I see a ${box.clsName} ${getDirectionLabel(box.cx)}, $dist away.", "DESCRIBE")
                }
            }
        }
    }

    private fun generateSceneDescription(boxes: List<BoundingBox>) {
        if (boxes.isEmpty()) speak("Nothing detected.", "SCENE_DESC")
        else speak("I see ${boxes.groupBy { it.clsName }.map { "${it.value.size} ${it.key}" }.joinToString(", ")}.", "SCENE_DESC")
    }
    
    private fun getDirectionLabel(cx: Float): String = when {
        cx < 0.38f -> "on your left"
        cx > 0.62f -> "on your right"
        else -> "ahead of you"
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.apply {
                language = Locale.US
                setSpeechRate(0.85f)
                setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(id: String?) { this@MainActivity.isSpeaking = true }
                    override fun onDone(id: String?) { this@MainActivity.isSpeaking = false; handleUtteranceDone(id) }
                    override fun onError(id: String?) { this@MainActivity.isSpeaking = false }
                })
            }
        }
    }
    
    private fun handleUtteranceDone(id: String?) {
        runOnUiThread {
            when (id) {
                "MANUAL_LISTEN" -> { currentState = AppState.LISTENING; startListening() }
                "REACHED_FINAL", "NAV_COMPLETE", "SCENE_DESC", "RESET_SWIPE", "COMPLETED", "ERROR_SILENT" -> {
                    currentState = AppState.DETECTING
                    binding.bottomNavigation.selectedItemId = R.id.navigation_camera
                }
            }
        }
    }
    
    private fun startListening() { try { speechRecognizer.startListening(recognitionIntent) } catch (e: Exception) {} }

    private var lastLowPrioritySpeakTime = 0L
    private val LOW_PRIORITY_COOLDOWN_MS = 2000L

    private fun speak(text: String, id: String) {
        val now = System.currentTimeMillis()

        // HIGH priority — cancel everything, speak NOW
        if (id == "HIGH_PRIORITY" || id == "REACHED_FINAL" || id == "OBSTACLE") {
            tts?.stop()
            isSpeaking = false
            val params = Bundle().apply { putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, id) }
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, id)
            lastSpeakTime = now
            return
        }

        // MEDIUM priority — queue it (don't silently drop)
        if (id == "NAV_GUIDANCE" || id == "NAV_CLOSE" || id == "NAV_LOST" || id == "NAV_FOUND") {
            if (now - lastLowPrioritySpeakTime < LOW_PRIORITY_COOLDOWN_MS) return
            val params = Bundle().apply { putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, id) }
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, id)
            lastSpeakTime = now
            lastLowPrioritySpeakTime = now
            return
        }

        // LOW priority — drop if speaking or too soon
        if (isSpeaking) return
        if (now - lastLowPrioritySpeakTime < LOW_PRIORITY_COOLDOWN_MS) return

        val params = Bundle().apply { putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, id) }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, id)
        lastSpeakTime = now
        lastLowPrioritySpeakTime = now
    }

    private fun toast(message: String) { runOnUiThread { Toast.makeText(baseContext, message, Toast.LENGTH_LONG).show() } }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }
    override fun onResume() { 
        super.onResume() 
        if (allPermissionsGranted()) startCamera() else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { if (it[Manifest.permission.CAMERA] == true && it[Manifest.permission.RECORD_AUDIO] == true) startCamera() }
    override fun onDestroy() { super.onDestroy() ; detector?.close() ; depthEstimator?.close() ; cameraExecutor.shutdown() ; try { speechRecognizer.destroy() } catch (e: Exception) {} ; tts?.stop() ; tts?.shutdown() }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event != null && event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accelX = event.values[0]
            accelY = event.values[1]
            accelZ = event.values[2]
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
    }
}
