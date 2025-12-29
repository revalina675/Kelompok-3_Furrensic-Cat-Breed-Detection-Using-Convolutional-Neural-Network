package com.example.furrensic

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var ivCat: ImageView
    private lateinit var placeholderLayout: LinearLayout
    private lateinit var tvBreedName: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var btnSelect: Button
    private lateinit var btnCamera: Button
    private lateinit var btnPredict: Button

    private var selectedBitmap: Bitmap? = null
    private lateinit var classifier: CatClassifier

    private var photoUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivCat = findViewById(R.id.ivCat)
        placeholderLayout = findViewById(R.id.placeholderLayout)
        tvBreedName = findViewById(R.id.tvBreedName)
        tvConfidence = findViewById(R.id.tvConfidence)
        btnSelect = findViewById(R.id.btnSelectImage)
        btnCamera = findViewById(R.id.btnCamera)
        btnPredict = findViewById(R.id.btnPredict)

        try {
            classifier = CatClassifier(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Gagal init model", Toast.LENGTH_LONG).show()
        }

        val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data: Intent? = result.data
                val imageUri: Uri? = data?.data
                if (imageUri != null) {
                    processImageUri(imageUri)
                }
            }
        }

        val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success && photoUri != null) {
                processImageUri(photoUri!!)
            }
        }

        btnSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            galleryLauncher.launch(intent)
        }

        btnCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                startCamera(cameraLauncher)
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
            }
        }


        btnPredict.setOnClickListener {
            if (selectedBitmap != null) {
                if (::classifier.isInitialized) {
                    val results = classifier.classify(selectedBitmap!!)
                    if (results.isNotEmpty()) {
                        val topResult = results[0]
                        val confidenceScore = topResult.confidence
                        val confidencePersen = confidenceScore * 100

                        if (confidenceScore < 0.50f) {
                            tvBreedName.text = "Tidak\nDikenali"
                            tvBreedName.setTextColor(Color.parseColor("#FFD54F"))
                            tvConfidence.text = "?"
                            tvConfidence.setTextColor(Color.parseColor("#FFD54F"))
                        } else {
                            var cleanLabel = topResult.label.replace("_", " ")
                            if (cleanLabel.contains(" ")) {
                                cleanLabel = cleanLabel.replaceFirst(" ", "\n")
                            }
                            tvBreedName.text = cleanLabel
                            tvBreedName.setTextColor(Color.WHITE)
                            tvConfidence.text = "%.2f%%".format(confidencePersen)
                            tvConfidence.setTextColor(Color.WHITE)
                        }
                    }
                }
            } else {
                Toast.makeText(this, "Pilih atau Foto dulu ya!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startCamera(launcher: androidx.activity.result.ActivityResultLauncher<Uri>) {
        val photoFile = createImageFile()

        photoUri = FileProvider.getUriForFile(
            this,
            "${applicationContext.packageName}.provider",
            photoFile
        )

        if (photoUri != null) {
            launcher.launch(photoUri!!)
        }
    }

    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

    private fun processImageUri(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val originalBitmap = BitmapFactory.decodeStream(inputStream)

            selectedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

            ivCat.setImageBitmap(selectedBitmap)
            placeholderLayout.visibility = View.GONE

            tvBreedName.text = "-"
            tvBreedName.setTextColor(Color.WHITE)
            tvConfidence.text = "0%"
            tvConfidence.setTextColor(Color.LTGRAY)

        } catch (e: Exception) {
            Toast.makeText(this, "Gagal memproses gambar", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::classifier.isInitialized) classifier.close()
    }
}