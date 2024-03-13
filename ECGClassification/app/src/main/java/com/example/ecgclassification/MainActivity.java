package com.example.ecgclassification;

import android.util.Pair;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
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

    private TextView classificationTextView;
    private List<String[]> csvData;
    private int currentRowIndex = -1;
    private LighterModel model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        classificationTextView = findViewById(R.id.classification);
        Button uploadButton = findViewById(R.id.uploadButton);

        classificationTextView.setText("Classification: Not Uploaded");

        csvData = readCSVFromAssets();

        try {
            model = LighterModel.newInstance(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "Error loading the model: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        uploadButton.setOnClickListener(v -> {
            Pair<ByteBuffer, String> inputData = loadInputData();
            ByteBuffer byteBuffer = inputData.first;
            String actualLabel = inputData.second;

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
                String predictedLabel = classes[maxIndex];

                String outputText = "Prediction: " + predictedLabel + "\nActual Label: " + actualClass;
                classificationTextView.setText(outputText);
            }
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
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
