package com.example.android.camerax.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Environment.DIRECTORY_MUSIC
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.android.example.camerax.tflite.databinding.ActivityCameraBinding
import com.example.android.camerax.tflite.faceDetection.FaceDetection
import com.example.android.camerax.tflite.tools.types.Detection
import com.example.android.camerax.tflite.tools.types.NMS
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageReference
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.io.FileWriter
import java.net.URL
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.collections.ArrayList
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

    private var pauseAnalysis = true
    private var imageRotationDegrees: Int = 0


    private val nnApiDelegate by lazy  {
        NnApiDelegate()
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate))
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
    fun faceDetection (){

        faceDetection.allocateTensors()

    }

    var eyeCor = Array(2){Array(2) { ArrayList<Float>() }}
    val predCor: FloatArray by lazy {
        val x = Math.random()
        val y = Math.random()
        floatArrayOf(0f, 0f)
    }
    val prevCor: FloatArray by lazy {
        floatArrayOf(0f, 0f)
    }
    var count = -1

    val dataGen = ArrayList<ArrayList<ArrayList<Float>>>()

    fun addData(eyeCor: Array<Array<ArrayList<Float>>>, predCor: FloatArray){
        //add to dataGen
        val eyeSpread = ArrayList<Float>()

        for(i in 0 until 2)
            for (j in 0 until eyeCor[i][0].size){
                eyeSpread.add(eyeCor[i][0][j])
                eyeSpread.add(eyeCor[i][1][j])
            }
        val dataNow = ArrayList<ArrayList<Float>>()
        dataNow.addAll(listOf(eyeSpread))
        val predList =ArrayList<Float>()
        for(i in 0 until predCor.size)
            predList.add(predCor[i])
        dataNow.addAll(listOf(predList))

        dataGen.add(dataNow)
    }


//    fun writeFileOnInternalStorage(mcoContext: Context, sFileName: String?, sBody: String?) {
//        val dir = File(mcoContext.getExternalFilesDir(DIRECTORY_MUSIC), "")
//        if (!dir.exists()) {
//            dir.mkdir()
//        }
//        try {
//            val gpxfile = File(dir, sFileName)
//            val writer = FileWriter(gpxfile)
//            writer.append(sBody)
//            writer.flush()
//            writer.close()
//        } catch (e: Exception) {
//            e.printStackTrace()
//        }
//    }
    var firstAtAll = true
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)

        setContentView(activityCameraBinding.root)
        faceDetection()
        faceLandmark.allocateTensors()

        Log.d(TAG,
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE).toString()
        )

        activityCameraBinding.saveButton!!.setOnClickListener{
//            writeFileOnInternalStorage(this,"proba.txt","tuka e datata")
            val reference = FirebaseStorage.getInstance().getReference().child("Document")
            val dataString = dataGen.joinToString("\n") { innerList ->
                innerList.joinToString("\n") { innerInnerList ->
                    innerInnerList.joinToString(" ") {
                        it.toString()
                    }
                }
            }
            dataGen.clear()
            val calendar = Calendar.getInstance()
            val day = calendar.get(Calendar.DAY_OF_MONTH)
            val hour = calendar.get(Calendar.HOUR_OF_DAY)
            val minute = calendar.get(Calendar.MINUTE)
            val second = calendar.get(Calendar.SECOND)
            val timeString = String.format("%02d-%02d-%02d-%02d", day, hour, minute, second)


            reference.child("${timeString}.txt").putBytes(dataString.toByteArray()).addOnSuccessListener{ Toast.makeText(this, "success", Toast.LENGTH_SHORT)}

        }
        var x = 0.0f
        var y = 0.0f
        activityCameraBinding.cameraCaptureButton.setOnClickListener {


            if(count!=-1){
                //call add
                addData(eyeCor,prevCor)
            }
            prevCor[0] = predCor[0]
            prevCor[1] = predCor[1]
            if(dataGen.size >0)
            if(dataGen[0][1][0]==0.0f){
                dataGen.clear()
            }
//            if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PERMISSION_GRANTED){}

            val fakedata = dataGen
            val fakePrev = prevCor
            count = dataGen.size
            activityCameraBinding.count!!.text = count.toString()
//            val splitY = Math.random()
//            var y: Float = 0f
//            if(splitY < 0.6)
//                y = Math.random().toFloat() * 0.36f + 0.6f
//            else
//                y= Math.random().toFloat() * 0.6f
//            val x: Float = Math.random().toFloat() * 0.96f
            if(x >= 0.91f){
                x = 0.00f
                y+= 0.09f
            } else x+= 0.07f
            predCor[0] = x
            predCor[1] = y
            (activityCameraBinding.circle!!.layoutParams as ViewGroup.MarginLayoutParams).apply {
                topMargin = (y*activityCameraBinding.viewFinder.height).toInt()
                leftMargin = (x*activityCameraBinding.viewFinder.width).toInt()

            }
            pauseAnalysis = false
            // Disable all camera controls

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

                var firstGetLandmarks = false
                var times = 0
                while(pauseAnalysis == false){

                if((firstGetLandmarks || firstAtAll) && times == 0) {
                    times ++
                    firstAtAll = false
                    faceDetection.runForMultipleInputsOutputs(arrayOf(newImage.buffer), outputMap)

//                transformFaceDetectionOutputs()

                    val helperFaceDetection = FaceDetection()
                    val boxes = helperFaceDetection.decodeBoxes(outputMap[0])
                    val scores = helperFaceDetection.getSigmoidScores(outputMap[1])
                    ddd = helperFaceDetection.convertToDetections(boxes, scores).toTypedArray()
                    val pruned = NMS.non_maximum_suppression(
                        ddd.toMutableList(), 0.3f,
                        0.5f // you can set this to whatever threshold you want
                        ,
                        true
                    )
                    val old = ddd
                    ddd = pruned.toTypedArray()
                }
                    firstGetLandmarks = true
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
                    val checkin = outputMapLandmark[1]!![0][0][0][0]
                    if(outputMapLandmark[1]!![0][0][0][0]>30f || times == 1){
                        pauseAnalysis = true

                    for(i in 0..1403    step 3) {
                        if(outputMapLandmark[0]!![0][0][0][i].toInt()<landmarkBitmap.width
                            && outputMapLandmark[0]!![0][0][0][i+1].toInt()<landmarkBitmap.height && outputMapLandmark[0]!![0][0][0][i].toInt()>0
                            && outputMapLandmark[0]!![0][0][0][i+1].toInt()>0){
                            landmarkBitmap.setPixel(
                                outputMapLandmark[0]!![0][0][0][i].toInt(),
                                outputMapLandmark[0]!![0][0][0][i+1].toInt(), color)
                        } else pauseAnalysis = false


                    }
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

//                    rotatedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, rotatedBitmap.width /3, rotatedBitmap.height /3, true)
//                    for(i in 0 until 2){
//                        for(j in 0 until 9){
//
    //                                rotatedBitmap.setPixel((eyeCor[i][0][j]).toInt(),(eyeCor[i][1][j]).toInt(),red)
//                        }
//                    }
//
//                    irisBitmaps[0] = rotatedBitmap
                    val fakedata = dataGen
                    reportTry(ddd[maxScoreIndex].data, outputMapLandmark[1]?.get(0)?.get(0)!![0][0], irisBitmaps)
                    }
                }
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
                    Log.d(TAG,  "${ddd.size}  ")
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

        activityCameraBinding.textPrediction.text = "${prevCor[0]}  ${eyeCor[0][1][0]}"

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
