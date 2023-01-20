package com.example.android.camerax.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.constraintlayout.widget.ConstraintSet
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.android.example.camerax.classLandmark.databinding.ActivityCameraBinding
import com.example.android.camerax.tflite.faceDetection.FaceDetection
import com.example.android.camerax.tflite.tools.types.Detection
import com.example.android.camerax.tflite.tools.types.NMS
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    lateinit var ddd: Array<Detection>
    private lateinit var activityCameraBinding: ActivityCameraBinding


    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer = TensorImage(DataType.UINT8)
    private val faceDetectionImageBuffer = TensorImage(DataType.FLOAT32)

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(
                128, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    private val nnApiDelegate by lazy  {
        NnApiDelegate()
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate))
    }
    private val detector by lazy {
        ObjectDetectionHelper(
            tflite,
            FileUtil.loadLabels(this, LABELS_PATH)
        )
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }
    // made for porcessing the image for face detection model
    private val tfImageFaceDetection by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(
                128, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0.5f, 0.5f))
            .build()
    }
    //this is the tflite model loaded
    private val faceDetection by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "face_detection_front.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    //this is the tflite model loaded Landmarks
    private val faceLandmark by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "face_landmark.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    //this is the tflite model loaded Landmarks
    private val irisLandmark by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "iris_landmark.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    private val corPos by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "xlite.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    private val YcorPos by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "ylite.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    //this is the output of the tflite model for face detection
     val outputMap: Map<Int, Array<Array<FloatArray>>> by lazy {
        val shape0: IntArray = faceDetection.getOutputTensor(0).shape()
        val output0 = Array(shape0[0]) { Array(shape0[1]) { FloatArray(shape0[2]) } }

        val shape1: IntArray = faceDetection.getOutputTensor(1).shape()
        val output1 = Array(shape1[0]) { Array(shape1[1]) { FloatArray(shape1[2]) } }

        mapOf(0 to output0, 1 to output1)
    }
    val outputMapLandmark: Map<Int, Array<Array<Array<FloatArray>>>> by lazy {
        val shape0: IntArray = faceLandmark.getOutputTensor(0).shape()
        val output0 = Array(shape0[0]) {Array(shape0[1]) { Array(shape0[2]) { FloatArray(shape0[3]) } }}

        val shape1: IntArray = faceLandmark.getOutputTensor(1).shape()
        val output1 = Array(shape1[0]) {Array(shape1[1]) { Array(shape1[2]) { FloatArray(shape1[3]) } }}

        mapOf(0 to output0, 1 to output1)
    }
    val outputMapIris: Map<Int, Array<FloatArray>> by lazy {
        val shape0: IntArray = irisLandmark.getOutputTensor(0).shape()
        val output0 = Array(shape0[0]) { FloatArray(shape0[1]) }

        val shape1: IntArray = irisLandmark.getOutputTensor(1).shape()
        val output1 =  Array(shape1[0]) { FloatArray(shape1[1]) }

        mapOf(0 to output0, 1 to output1)
    }

    var eyeCor = Array(2){Array(2) { ArrayList<Float>() }}


    fun faceDetection (){

        faceDetection.allocateTensors()

    }

    var detections: Array<Detection> = arrayOf()

//    fun transformFaceDetectionOutputs() {
//        val helperFaceDetection = FaceDetection()
//        val boxes = helperFaceDetection.decodeBoxes(outputMap[0])
//        val scores = helperFaceDetection.getSigmoidScores(outputMap[1])
//        detections = helperFaceDetection.convertToDetections(boxes, scores).toTypedArray()
//    }


    lateinit var square: View

    var firstView = true
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)

        setContentView(activityCameraBinding.root)
        faceDetection()
        faceLandmark.allocateTensors()
//        ConstraintLayout.LayoutParams(activityCameraBinding.circle!!.layoutParams).horizontalBias = 50f
//        activityCameraBinding.circle!!.layoutParams = ConstraintLayout.LayoutParams(activityCameraBinding.circle!!.layoutParams)
//        activityCameraBinding.circle!!.setHorizontalBias(0.7)
//        activityCameraBinding.circle!!.setVerticalBias(0.7)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {
            if(firstView){
            square = View(this)
            square.setBackgroundColor(Color.argb(127, 255, 0, 0))
            square.setId(View.generateViewId());
                val layoutParams = ConstraintLayout.LayoutParams(activityCameraBinding.viewFinder.width/2, activityCameraBinding.viewFinder.height/4)
                square.layoutParams = layoutParams
                activityCameraBinding.root.addView(square)
                val constraintSet = ConstraintSet()
                constraintSet.clone(activityCameraBinding.root)
                constraintSet.connect(
                    square.id,
                    ConstraintSet.START,
                    ConstraintSet.PARENT_ID,
                    ConstraintSet.START
                )
                constraintSet.connect(
                    square.id,
                    ConstraintSet.TOP,
                    ConstraintSet.PARENT_ID,
                    ConstraintSet.TOP
                )
                constraintSet.applyTo(activityCameraBinding.root)


                firstView = false

//


            }
//            val splitY = Math.random()
//            var y: Float = 0f
//            if(splitY < 0.6)
//                y = Math.random().toFloat() * 0.4f + 0.6f
//            else
//                y= Math.random().toFloat() * 0.4f
//            val x: Float = Math.random().toFloat()
//
//            (activityCameraBinding.circle!!.layoutParams as ViewGroup.MarginLayoutParams).apply {
//                topMargin = (y*activityCameraBinding.viewFinder.height).toInt()
//                leftMargin = (x*activityCameraBinding.viewFinder.width).toInt()
//
//            }
//            pauseAnalysis = false
            // Disable all camera controls
            it.isEnabled = false

            //set circle
//            val redDot: ImageView = findViewById(R.id.circle)

            if (pauseAnalysis) {
                // If image analysis is in paused state, resume it
                pauseAnalysis = false
                activityCameraBinding.imagePredicted.visibility = View.GONE

            } else {
                // Otherwise, pause image analysis and freeze image
                pauseAnalysis = true
                val matrix = Matrix().apply {
                    postRotate(imageRotationDegrees.toFloat())
                    if (isFrontFacing) postScale(-1f, 1f)
                }
                val uprightImage = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
                activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)

                activityCameraBinding.imagePredicted.visibility = View.VISIBLE
            }

            // Re-enable camera controls
            it.isEnabled = true
        }
    }

    override fun onDestroy() {

        // Terminate all outstanding analyzing jobs (if there is any).
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }

        // Release TFLite resources.
        tflite.close()
        nnApiDelegate.close()

        super.onDestroy()
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError", "UnsafeOptInUsageError")

    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener ({

            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()


            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                // Early exit: image analysis is in paused state
                if (pauseAnalysis) {
                    image.close()
                    return@Analyzer
                }
//                pauseAnalysis = true
//                if()
                // Copy out RGB bits to our shared buffer
                image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)  }
        //new paln
                var matrix = Matrix()
                matrix.postRotate(-90f)
                var rotatedBitmap = Bitmap.createBitmap(
                    bitmapBuffer,
                    0,
                    0,
                    bitmapBuffer.width,
                    bitmapBuffer.height,
                    matrix,
                    true
                )

                // Resize Bitmap to 128x128
                val resizedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, 128, 128, true)

                // Rotate Bitmap by -90 degrees


                val width: Int = resizedBitmap.getWidth()
                val height: Int = resizedBitmap.getHeight()


                val fpixels = FloatArray(width * height * 3)
                var counter = 0

                for (x in 0..width-1) {
                    for (y in 0..height-1) {
                        val pixel: Int = resizedBitmap.getPixel(x, y)

                        // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                        // model. For example, some models might require values to be normalized
                        // to the range [0.0, 1.0] instead.
                        fpixels[counter]= Color.red(pixel) / 255.0f
                        fpixels[counter+1]= Color.green(pixel) / 255.0f
                        fpixels[counter+2]= Color.blue(pixel) / 255.0f

                        fpixels[counter] = (fpixels[counter]- 0.5f) / 0.5f
                        fpixels[counter+1] = (fpixels[counter+1]- 0.5f) / 0.5f
                        fpixels[counter+2] = (fpixels[counter+2]- 0.5f) / 0.5f

                        counter += 3
                    }
                }
                val newImage = TensorImage(DataType.FLOAT32)
                val myArr: IntArray = intArrayOf(1,128,128,3)
                newImage.apply { load(fpixels, myArr) }

                if(outputMapLandmark[1]!![0][0][0][0]<42f){
                faceDetection.runForMultipleInputsOutputs(arrayOf(newImage.buffer),outputMap )

//                transformFaceDetectionOutputs()

                val helperFaceDetection = FaceDetection()
                val boxes = helperFaceDetection.decodeBoxes(outputMap[0])
                val scores = helperFaceDetection.getSigmoidScores(outputMap[1])
                ddd = helperFaceDetection.convertToDetections(boxes, scores).toTypedArray()
                val pruned = NMS.non_maximum_suppression(ddd.toMutableList(),  0.3f,
                    0.5f // you can set this to whatever threshold you want
                    ,
                    true)
                    val old = ddd
                ddd = pruned.toTypedArray()
                }

                if(ddd.size > 0){
//                    var maxScore = -1f
//                    var maxScoreIndex = -1
//                    for (i in ddd.indices) {
//                        if (ddd[i].score > maxScore) {
//                            maxScore = ddd[i].score
//                            maxScoreIndex = i
//                        }
//                    }

                    var maxScoreIndex = 0
//                    if(prevX == -1){
//                        prevX = (ddd[maxScoreIndex].data[1]*rotatedBitmap.width ).toInt()
//                        prevY = (ddd[maxScoreIndex].data[0]*rotatedBitmap.height ).toInt()
//                    }
//                    var has = false
//                    for (i in ddd.indices) {
//                        if((ddd[maxScoreIndex].data[1]*rotatedBitmap.width ).toInt()== prevX &&
//                            prevY == (ddd[maxScoreIndex].data[0]*rotatedBitmap.height ).toInt()) {
//                            maxScoreIndex = i
//                            has = true
//                        }
//                    }
//                    if(!has){
//                        prevX = (ddd[maxScoreIndex].data[1]*rotatedBitmap.width ).toInt()
//                        prevY = (ddd[maxScoreIndex].data[0]*rotatedBitmap.height ).toInt()
//                    }
                //here we do face landmark

                    var sizedX =((ddd[maxScoreIndex].data[1]*5-ddd[maxScoreIndex].data[3])*rotatedBitmap.width/4).toInt()
                    var sizedY =((ddd[maxScoreIndex].data[0]*5-ddd[maxScoreIndex].data[2])*rotatedBitmap.height/4).toInt()
                    var sizedWidth = ((ddd[maxScoreIndex].data[3]-ddd[maxScoreIndex].data[1])*rotatedBitmap.width*1.5f).toInt()
                    var sizedHeight = ((ddd[maxScoreIndex].data[2]-ddd[maxScoreIndex].data[0])*rotatedBitmap.height*1.5f).toInt()

                    sizedX = when(sizedX > 0) {
                        true -> sizedX
                        false -> 0
                    }
                    sizedX = when(sizedX < rotatedBitmap.width) {
                        true -> sizedX
                        false -> rotatedBitmap.width-1
                    }
                    sizedY = when(sizedY > 0) {
                        true -> sizedY
                        false -> 0
                    }
                    sizedY = when(sizedY < rotatedBitmap.height) {
                        true -> sizedY
                        false -> rotatedBitmap.height-1
                    }
                    sizedWidth = when(sizedX+sizedWidth < rotatedBitmap.width) {
                        true -> sizedWidth
                        false -> rotatedBitmap.width-sizedX
                    }
                    sizedHeight = when(sizedY+sizedHeight < rotatedBitmap.height) {
                        true -> sizedHeight
                        false -> rotatedBitmap.height-sizedY
                    }

                    var landmarkBitmap = Bitmap.createBitmap(rotatedBitmap,sizedX,sizedY,
                            sizedWidth, sizedHeight);
                    landmarkBitmap = Bitmap.createScaledBitmap(landmarkBitmap,192,192,true);
                    val color = Color.RED
                    val lpixels = FloatArray(192 * 192 * 3)
                    var counterl = 0
                    for (x in 0 until 192) {
                        for (y in 0 until 192) {
                            val pixel: Int = landmarkBitmap.getPixel(y, x)

                            // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                            // model. For example, some models might require values to be normalized
                            // to the range [0.0, 1.0] instead.
//                            lpixels[counterl]= Color.red(pixel).toFloat()
//                            lpixels[counterl+1]= Color.green(pixel).toFloat()
//                            lpixels[counterl+2]= Color.blue(pixel).toFloat()
                            lpixels[counterl]= Color.red(pixel)/255f
                            lpixels[counterl+1]= Color.green(pixel)/255f
                            lpixels[counterl+2]= Color.blue(pixel)/255f

                            counterl += 3
                        }
                    }



                    val newImageL = TensorImage(DataType.FLOAT32)
                    val myArrL: IntArray = intArrayOf(1,192,192,3)
                    newImageL.apply { load(lpixels, myArrL) }

                    faceLandmark.runForMultipleInputsOutputs(arrayOf(newImageL.buffer),outputMapLandmark )
                    val proba = outputMapLandmark
//                    for(i in 0..1403    step 3) {
//                        if(outputMapLandmark[0]!![0][0][0][i].toInt()<landmarkBitmap.width
//                            && outputMapLandmark[0]!![0][0][0][i+1].toInt()<landmarkBitmap.height)
//                        landmarkBitmap.setPixel(
//                            outputMapLandmark[0]!![0][0][0][i].toInt(),
//                            outputMapLandmark[0]!![0][0][0][i+1].toInt(), color
//                        )
//                    }
//                   //33, 133
                    //, 362, 263
                    val numbers = intArrayOf(33, 133,362, 263)
                    var irisBitmaps = arrayOfNulls<Bitmap>(2)
                    eyeCor = Array(2){Array(2) { ArrayList<Float>() }}
                    for (num in 0 until numbers.size step 2) {
                        val left = generateFloatArray(numbers[num])
                        val right = generateFloatArray(numbers[num+1])
//                                                landmarkBitmap.setPixel(
//                            outputMapLandmark[0]!![0][0][0][floatArr[0].toInt()].toInt(),
//                            outputMapLandmark[0]!![0][0][0][floatArr[1].toInt()].toInt(), color
                        left[0] = outputMapLandmark[0]!![0][0][0][left[0].toInt()]/192f
                        left[1] = outputMapLandmark[0]!![0][0][0][left[1].toInt()]/192f
                        left[2] = outputMapLandmark[0]!![0][0][0][left[2].toInt()]/192f
                        right[0] = outputMapLandmark[0]!![0][0][0][right[0].toInt()]/192f
                        right[1]  = outputMapLandmark[0]!![0][0][0][right[1].toInt()]/192f
                        right[2]  = outputMapLandmark[0]!![0][0][0][right[2].toInt()]/192f
                        var middleX = (right[0]+left[0])/2 * sizedWidth + sizedX
                        var middleY = (right[1]+left[1])/2  * sizedHeight + sizedY

                        var  IsizedWidth = (right[0]-left[0])*2.3f * sizedWidth
                        var  IsizedHeight =IsizedWidth
                        var IsizedX = middleX - IsizedWidth/2
                        var  IsizedY = middleY - IsizedHeight/2

                        IsizedX = when(IsizedX > 0) {
                            true -> IsizedX
                            false -> 0f
                        }
                        IsizedX = when(IsizedX < rotatedBitmap.width) {
                            true -> IsizedX
                            false -> rotatedBitmap.width-1f
                        }
                        IsizedY = when(IsizedY > 0) {
                            true -> IsizedY
                            false -> 0f
                        }
                        IsizedY = when(IsizedY < rotatedBitmap.height) {
                            true -> IsizedY
                            false -> rotatedBitmap.height-1f
                        }
                        IsizedWidth = when(IsizedX+IsizedWidth < rotatedBitmap.width) {
                            true -> IsizedWidth
                            false -> rotatedBitmap.width-IsizedX
                        }
                        IsizedHeight = when(IsizedY+IsizedHeight < rotatedBitmap.height) {
                            true -> IsizedHeight
                            false -> rotatedBitmap.height-IsizedY
                        }


                        var irisBitmap = Bitmap.createBitmap(rotatedBitmap,IsizedX.toInt(),IsizedY.toInt(),
                            IsizedWidth.toInt(), IsizedHeight.toInt());
                        irisBitmap = Bitmap.createScaledBitmap(irisBitmap,64,64,true);
                       if(num == 2){
                           val matrix = Matrix()
                           matrix.preScale(-1.0f, 1.0f)
                           irisBitmap = Bitmap.createBitmap(
                               irisBitmap,
                               0,
                               0,
                               irisBitmap.getWidth(),
                               irisBitmap.getHeight(),
                               matrix,
                               true
                           )
                       }
                        val ipixels = FloatArray(64 * 64 * 3)
                        var counteri = 0
                        for (x in 0 until 64) {
                            for (y in 0 until 64) {
                                val pixel: Int = irisBitmap.getPixel(y, x)
                                ipixels[counteri]= Color.red(pixel)/255f
                                ipixels[counteri+1]= Color.green(pixel)/255f
                                ipixels[counteri+2]= Color.blue(pixel)/255f

                                counteri += 3
                            }
                        }
                        val newImageI = TensorImage(DataType.FLOAT32)
                        val myArrI: IntArray = intArrayOf(1,64,64,3)
                        newImageI.apply { load(ipixels, myArrI) }

                        irisLandmark.runForMultipleInputsOutputs(arrayOf(newImageI.buffer),outputMapIris)

                        val irisParts = intArrayOf(0,4,8,12)
                        eyeCor[num/2][0].add( IsizedX / rotatedBitmap.width )
                        eyeCor[num/2][1].add( IsizedY / rotatedBitmap.height)
                        eyeCor[num/2][0].add( IsizedWidth / rotatedBitmap.width)
                        eyeCor[num/2][1].add( IsizedHeight / rotatedBitmap.height)
                        for (b in 0 until irisParts.size) {
                            val cor = generateFloatArray(irisParts[b])
                            irisBitmap.setPixel(
                                outputMapIris[0]!![0][cor[0].toInt()].toInt(),
                                outputMapIris[0]!![0][cor[1].toInt()].toInt(), color
                            )
                            //get cor
                            eyeCor[num/2][0].add( outputMapIris[0]!![0][cor[0].toInt()]/64 )
                            eyeCor[num/2][1].add( outputMapIris[0]!![0][cor[1].toInt()]/64 )
                        }
                        for(i in 0 until 15    step 3) {
                            if(outputMapIris[1]!![0][i].toInt()<irisBitmap.width
                                && outputMapIris[1]!![0][i+1].toInt()<irisBitmap.height) {
                                irisBitmap.setPixel(
                                    outputMapIris[1]!![0][i].toInt(),
                                    outputMapIris[1]!![0][i + 1].toInt(), color

                                )
                                //cor
                                eyeCor[num/2][0].add( outputMapIris[1]!![0][i]/64 )
                                eyeCor[num/2][1].add( outputMapIris[1]!![0][i+1]/64 )
                            }
                        }
                        if(num==0){
                            val matrix = Matrix()
                            matrix.preScale(-1.0f, 1.0f)
                            irisBitmap = Bitmap.createBitmap(
                                irisBitmap,
                                0,
                                0,
                                irisBitmap.getWidth(),
                                irisBitmap.getHeight(),
                                matrix,
                                true
                            )
                        }
                        irisBitmaps[num/2] = irisBitmap


                    }
                    var corX = 0.0f
                    var corY = 0.0f
                    for( i in 0 until 2){
                        val eyeSpread = ArrayList<Float>()

                        for (i in 0 until 2)
                            for (j in 0 until eyeCor[i][0].size) {
                                eyeSpread.add(eyeCor[i][0][j])
                                eyeSpread.add(eyeCor[i][1][j])
                            }
                        val dataNow = ArrayList<ArrayList<Float>>()
                        dataNow.addAll(listOf(eyeSpread))
                        val corOutput = Array(1) { FloatArray(8) }
                        val myFloatArray = dataNow[0].toFloatArray()

                        val myByteBuffer = ByteBuffer.allocateDirect(44 * 4) // 4 bytes per float
                            .order(ByteOrder.nativeOrder())
                            .asFloatBuffer()
                            .put(myFloatArray)
//                        if(i == 0) {
//                            corPos.run(myByteBuffer, corOutput)
//                            var index = 0
//                            for(i in 0 until 15){
//                                if(corOutput[0][i]>corOutput[0][index]){
//                                    index = i
//                                }
//                            }
//                            corX = (index*7)/100f
//
//                        }else {
//                            YcorPos.run(myByteBuffer, corOutput)
//                            var index = 0
//                            for(i in 0 until 15){
//                                if(corOutput[0][i]>corOutput[0][index]){
//                                    index = i
//                                }
//                            }
//                            corY = (index*7)/100f
//                        }



                        if(!firstView){


                            corPos.run(myByteBuffer, corOutput)
                            val copie = corOutput
                            var index = 0
                            for(i in 0 until corOutput[0].size){
                            if(corOutput[0][i]>corOutput[0][index]){
                                index=i
                            }
                        }
                        (square.layoutParams as ViewGroup.MarginLayoutParams).apply {
                            topMargin = ((index/2)*0.25*activityCameraBinding.viewFinder.height).toInt()
                            leftMargin = ((index%2)*0.5*activityCameraBinding.viewFinder.width).toInt()

                        }}

                    }
                    (activityCameraBinding.circle!!.layoutParams as ViewGroup.MarginLayoutParams).apply {
                        topMargin = (corX*activityCameraBinding.viewFinder.height).toInt()
                        leftMargin = (corX*activityCameraBinding.viewFinder.width).toInt()

                    }



                    reportTry(ddd[maxScoreIndex].data, outputMapLandmark[1]?.get(0)?.get(0)!![0][0], irisBitmaps)


//                    android.setImageBitmap(landmarkBitmap)

                }



                // Perform the object detection for the current frame
//                val predictions = detector.predict(tfImage)

                // Report only the top prediction
//                reportPrediction(predictions.maxByOrNull { it.score })

                // Compute the FPS of the entire pipeline
                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    val fps = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG,  "${ddd.size}" )
//                    Log.d(TAG, "FPS: ${"%.02f".format(fps)} with tensorSize: ${tfImage.width} x ${tfImage.height}")
                    lastFpsTimestamp = now
                }

            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this))
    }

//        fun addData(eyeCor: Array<Array<ArrayList<Float>>>){
//            //add to dataGen
//            val eyeSpread = ArrayList<Float>()
//
//            for(i in 0 until 2)
//                for (j in 0 until eyeCor[i][0].size){
//                    eyeSpread.add(eyeCor[i][0][j])
//                    eyeSpread.add(eyeCor[i][1][j])
//                }
//            val dataNow = ArrayList<ArrayList<Float>>()
//            dataNow.addAll(listOf(eyeSpread))
//            val dataString = dataNow.joinToString(" ") {
//                it.toString()
//
//        }
//
//        }

    fun generateFloatArray(num: Int): FloatArray {
        return floatArrayOf(num*3f, ((num*3)+1f), ((num*3)+2f))
    }
    //created as reportPrediction but for face detection
//    private fun reportFace(prediction: FloatArray) = activityCameraBinding.viewFinder.post {
//
//
//        // Location has to be mapped to our local coordinates
//        val floatArray = floatArrayOf(prediction[0], prediction[1],prediction[2], prediction[3])
//
//
//        // Update the text and UI
//        activityCameraBinding.textPrediction.text = "${floatArray[0]},${floatArray[1]},${floatArray[2]},${floatArray[3]}"
//        (activityCameraBinding.boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
//            topMargin = floatArray[0].toInt()
//            leftMargin = floatArray[1].toInt()
//            width = min(activityCameraBinding.viewFinder.width, floatArray[3].toInt() - floatArray[1].toInt())
//            height = min(activityCameraBinding.viewFinder.height, floatArray[2].toInt() - floatArray[0].toInt())
//        }
//
//
//        // Make sure all UI elements are visible
//        activityCameraBinding.boxPrediction.visibility = View.VISIBLE
//        activityCameraBinding.textPrediction.visibility = View.VISIBLE
//    }
    private fun reportTry(
        prediction: FloatArray, score: Float, landmarkBitmap: Array<Bitmap?>
    ) = activityCameraBinding.viewFinder.post {



        // Location has to be mapped to our local coordinates
        val rectF = RectF()
        rectF.top = prediction[0]
        rectF.left = prediction[1]
        rectF.bottom = prediction[2]
        rectF.right = prediction[3]
        val location = mapOutputCoordinates(rectF)

        // Update the text and UI
//        activityCameraBinding.textPrediction.text = "${location.top},${location.left},${location.bottom},${location.right}"
        activityCameraBinding.textPrediction.text = "${score}"

        (activityCameraBinding.boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
            height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
        }



        // Make sure all UI elements are visible
        activityCameraBinding.landmarks!!.setImageBitmap(landmarkBitmap[1])

        activityCameraBinding.landmarks2!!.setImageBitmap(landmarkBitmap[0])
        activityCameraBinding.boxPrediction.visibility = View.VISIBLE
        activityCameraBinding.textPrediction.visibility = View.VISIBLE
    }

    /**
     * Helper function used to map the coordinates for objects coming out of
     * the model into the coordinates that the user sees on the screen.
     */
    private fun mapOutputCoordinates(location: RectF): RectF {

        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            location.left * activityCameraBinding.viewFinder.width,
            location.top * activityCameraBinding.viewFinder.height,
            location.right * activityCameraBinding.viewFinder.width,
            location.bottom * activityCameraBinding.viewFinder.height
        )

        // Step 2: compensate for camera sensor orientation and mirroring
        val isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT
        val correctedLocation = if (isFrontFacing) {
            RectF(
                activityCameraBinding.viewFinder.width - previewLocation.right,
                previewLocation.top,
                activityCameraBinding.viewFinder.width - previewLocation.left,
                previewLocation.bottom)
        } else {
            previewLocation
        }

        // Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
        val margin = 0.1f
        val requestedRatio = 4f / 3f
        val midX = (correctedLocation.left + correctedLocation.right) / 2f
        val midY = (correctedLocation.top + correctedLocation.bottom) / 2f
        return if (activityCameraBinding.viewFinder.width < activityCameraBinding.viewFinder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY - (1f - margin) * correctedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY + (1f - margin) * correctedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * correctedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * correctedLocation.height() / 2f,
                midX + (1f - margin) * correctedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * correctedLocation.height() / 2f
            )
        }
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val ACCURACY_THRESHOLD = 0.5f
        private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }
}
