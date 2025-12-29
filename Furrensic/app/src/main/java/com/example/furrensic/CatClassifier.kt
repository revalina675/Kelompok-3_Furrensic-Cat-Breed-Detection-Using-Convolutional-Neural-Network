package com.example.furrensic

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class CatClassifier(context: Context) {

    private val modelName = "model_resnet_final.tflite"
    private val labelName = "labels.txt"
    private val inputSize = 224

    private val mean = floatArrayOf(123.68f, 116.78f, 103.94f)
    private val std = floatArrayOf(1.0f, 1.0f, 1.0f)

    private val model: Model
    private val labels: List<String>
    private val imageProcessor: ImageProcessor

    init {
        model = Model.createModel(context, modelName)
        labels = FileUtil.loadLabels(context, labelName)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(mean, std))
            .build()
    }

    fun classify(bitmap: Bitmap): List<ClassificationResult> {

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 12), DataType.FLOAT32)

        val inputs = arrayOf(tensorImage.buffer)
        val outputs = mutableMapOf<Int, Any>(0 to outputBuffer.buffer)

        model.run(inputs, outputs)

        val labeledProbability = TensorLabel(labels, outputBuffer).mapWithFloatValue

        return labeledProbability.entries
            .map { ClassificationResult(it.key, it.value) }
            .sortedByDescending { it.confidence }
    }

    data class ClassificationResult(val label: String, val confidence: Float)

    fun close() {
        model.close()
    }
}