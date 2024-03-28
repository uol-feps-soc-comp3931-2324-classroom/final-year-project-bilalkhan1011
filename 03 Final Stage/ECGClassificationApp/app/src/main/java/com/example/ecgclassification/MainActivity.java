package com.example.ecgclassification;

import android.os.Handler;
import android.util.Pair;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.view.View;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.ecgclassification.ml.LighterModel;
import com.opencsv.CSVReader;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.DataType;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private List<String[]> csvData;
    private int currentRowIndex = -1;
    private LighterModel model;
    private Handler handler = new Handler();
    private Runnable runnableCode;
    private int abnormalBeatsCount = 0;
    private TextView predictionTextView; // Assume this TextView exists in your layout
    private ImageView statusImageView; // Assume this ImageView exists in your layout

    private TextView additionalText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        predictionTextView = findViewById(R.id.predictionTextView);
        statusImageView = findViewById(R.id.statusImageView);
        additionalText = findViewById(R.id.additionalText);

        csvData = readCSVFromAssets();

        Pair<ByteBuffer, String> inputData = loadInputData();
        ByteBuffer byteBuffer = inputData.first;
        String actualLabel = inputData.second;

        try {
            model = LighterModel.newInstance(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "Error loading the model: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        // Define the code block to be executed
        runnableCode = new Runnable() {
            @Override
            public void run() {
                // Insert the code from your if(byteBuffer != null) { ... } block here
                String predictedLabel = null;
                if (byteBuffer != null) {
                    // Ensure buffer uses the correct byte order
                    byteBuffer.order(ByteOrder.nativeOrder());
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 2160, 1}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    LighterModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] outputArray = outputFeature0.getFloatArray();

                    int maxIndex = 0;
                    for (int i = 1; i < outputArray.length; i++) {
                        if (outputArray[i] > outputArray[maxIndex]) {
                            maxIndex = i;
                        }
                    }

                    String[] classes = {"Normal", "Ventricular", "Supraventricular", "Fusion", "Other"};
                    String actualClass = classes[Integer.parseInt(actualLabel)];
                    predictedLabel = classes[maxIndex];

                    String outputText = "Prediction: " + predictedLabel + "\nActual Label: " + actualClass;
                }

                // Check the predicted label and update the abnormalBeatsCount accordingly
                if (predictedLabel.equals("Other") || predictedLabel.equals("Normal")) {
                    abnormalBeatsCount = 0; // Reset if the beat is normal or other
                } else {
                    abnormalBeatsCount++;
                }

                // Update UI based on the count of abnormal beats
                updateUI(abnormalBeatsCount);

                // Repeat this runnable code block again every 5 seconds
                handler.postDelayed(this, 5000);
            }
        };

        // Start the initial runnable task by posting through the handler
        handler.post(runnableCode);
    }


    private void updateUI(int abnormalBeatsCount) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (abnormalBeatsCount >= 5) {
                    statusImageView.setImageResource(R.drawable.orange); // Change the drawable to indicate a warning
                    predictionTextView.setText(R.string.orange);
                    additionalText.setText(R.string.orangetext);
                }
                if (abnormalBeatsCount >= 10) {
                    statusImageView.setImageResource(R.drawable.red); // Change the drawable to indicate danger
                    predictionTextView.setText(R.string.red);
                    additionalText.setText(R.string.redtext);
                }
            }
        });
    }


    @Override
    protected void onPause() {
        super.onPause();
        if (model != null) {
            model.close();
            model = null;
        }
    }

    private Pair<ByteBuffer, String> loadInputData() {
        if (csvData == null || csvData.isEmpty()) return new Pair<>(null, "No Data");

        int numRows = csvData.size();
        currentRowIndex = (currentRowIndex + 1) % numRows;
        String[] selectedRow = csvData.get(currentRowIndex);
        String label = selectedRow[selectedRow.length - 1];

        int numCols = selectedRow.length - 1;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(numCols * Float.BYTES);
        byteBuffer.order(ByteOrder.nativeOrder()); // Ensure the buffer uses the system's native byte order

        for (int i = 0; i < numCols; i++) {
            byteBuffer.putFloat(Float.parseFloat(selectedRow[i]));
        }
        byteBuffer.rewind();

        return new Pair<>(byteBuffer, label);
    }

    private List<String[]> readCSVFromAssets() {
        AssetManager assetManager = getAssets();
        List<String[]> csvData = new ArrayList<>();
        try {
            InputStream inputStream = assetManager.open("data.csv");
            CSVReader reader = new CSVReader(new InputStreamReader(inputStream));
            List<String[]> allRows = reader.readAll();

            // Check if the list is not empty and has more than one row
            if (!allRows.isEmpty()) {
                // Skip the first row (header) and add the rest to csvData
                csvData.addAll(allRows.subList(1, allRows.size()));
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return csvData;
    }
}
