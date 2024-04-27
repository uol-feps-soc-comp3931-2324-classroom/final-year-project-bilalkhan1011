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

    // prepare variables for use in the application
    private List<String[]> csvData;
    private int currentRowIndex = -1;
    private LighterModel model;
    private Handler handler = new Handler();
    private Runnable runnableCode;
    private int abnormalBeatsCount = 0;
    private int unknownBeatsCount = 0;
    private TextView predictionTextView;
    private ImageView statusImageView;

    private TextView additionalText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // find relevant elements by their ID's to set them as variables for updating
        predictionTextView = findViewById(R.id.predictionTextView);
        statusImageView = findViewById(R.id.statusImageView);
        additionalText = findViewById(R.id.additionalText);

        // read in the sample data required for demo
        csvData = readCSVFromAssets();

        // try to create the deep learning model using the file
        try {
            model = LighterModel.newInstance(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "Error loading the model: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        // create a new section of runnable code
        runnableCode = new Runnable() {
            @Override
            public void run() {

                // load in the data per row and prepare it for processing
                Pair<ByteBuffer, String> inputData = loadInputData();
                ByteBuffer byteBuffer = inputData.first;
                String actualLabel = inputData.second;

                // prepare the data for processing by the model
                if (byteBuffer != null) {
                    byteBuffer.order(ByteOrder.nativeOrder());
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 2160, 1}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // feed data to model and collect the output
                    LighterModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] outputArray = outputFeature0.getFloatArray();

                    int maxIndex = 0;
                    for (int i = 1; i < outputArray.length; i++) {
                        if (outputArray[i] > outputArray[maxIndex]) {
                            maxIndex = i;
                        }
                    }

                    // categorise the output using the classes pre-set in the AAMI standard
                    String[] classes = {"Normal", "Ventricular", "Supraventricular", "Fusion", "Other"};
                    String actualClass = classes[Integer.parseInt(actualLabel)];
                    String predictedLabel = classes[maxIndex];

                    // update the counter variables and UI
                    updateCounts(predictedLabel);
                    updateUI(abnormalBeatsCount, unknownBeatsCount);

                    handler.postDelayed(this, 3000);
                }
            }
        };

        handler.post(runnableCode);
    }

    // update each of the counter variables depending on the label classified by the model
    private void updateCounts(String predictedLabel) {
        if (predictedLabel.equals("Other")) {
            unknownBeatsCount++;
        } else {
            abnormalBeatsCount++;
        }
    }

    // update the interface depending on the classified labels
    private void updateUI(int abnormalBeatsCount, int unknownBeatsCount) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {

                // low threshold will lead to all of the ECG summary elements to update to orange
                if (abnormalBeatsCount >= 5) {
                    statusImageView.setImageResource(R.drawable.orange);
                    predictionTextView.setText(R.string.orange);
                    additionalText.setText(R.string.orangetext);
                }

                // high threshold, more serious instructions
                if (abnormalBeatsCount >= 6) {
                    statusImageView.setImageResource(R.drawable.red);
                    predictionTextView.setText(R.string.red);
                    additionalText.setText(R.string.redtext);
                }

                //unknown readings, advises to readjust device
                if (unknownBeatsCount >= 10) {
                    statusImageView.setImageResource(R.drawable.grey);
                    predictionTextView.setText(R.string.grey);
                    additionalText.setText(R.string.greytext);
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

    // load in the CSV file provided for sample data containing diverse examples of each beat
    private Pair<ByteBuffer, String> loadInputData() {

        if (csvData == null || csvData.isEmpty()) return new Pair<>(null, "No Data");

        // set the array up for handling the ECG data and load in the true labels
        int numRows = csvData.size();
        currentRowIndex = (currentRowIndex + 1) % numRows;
        String[] selectedRow = csvData.get(currentRowIndex);
        String label = selectedRow[selectedRow.length - 1];

        // turn the data into Byte Buffers suitable for feeding into the model
        int numCols = selectedRow.length - 1;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(numCols * Float.BYTES);
        byteBuffer.order(ByteOrder.nativeOrder());

        for (int i = 0; i < numCols; i++) {
            byteBuffer.putFloat(Float.parseFloat(selectedRow[i]));
        }
        byteBuffer.rewind();

        return new Pair<>(byteBuffer, label);

    }

    // read CSV from the assets folder
    private List<String[]> readCSVFromAssets() {

        // find the csv file
        AssetManager assetManager = getAssets();
        List<String[]> csvData = new ArrayList<>();

        // open the file and begin reading it
        try {
            InputStream inputStream = assetManager.open("data.csv");
            CSVReader reader = new CSVReader(new InputStreamReader(inputStream));
            List<String[]> allRows = reader.readAll();

            if (!allRows.isEmpty()) {
                csvData.addAll(allRows.subList(1, allRows.size()));
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // returns the csv data as an list
        return csvData;
    }
}
