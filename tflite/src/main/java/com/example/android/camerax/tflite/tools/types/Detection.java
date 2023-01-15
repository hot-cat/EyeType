package com.example.android.camerax.tflite.tools.types;

import androidx.annotation.NonNull;

import java.util.Arrays;

public class Detection {
    public float score;
    public float[] data;


    public Detection(float[][] data, float score) {
        int rows = data.length;
        int cols = data[0].length;
        this.data = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data, i * cols, cols);
        }
        this.score = score;
    }
    public Detection(float[] data, float score) {
        this.data = data;
        this.score = score;
    }

    @NonNull
    @Override
    public String toString() {
        String dataString = Arrays.toString(data);
        return "Detection [data=" + dataString + ", score=" + score + "]";
    }
//    public float[] getKeypoint(int key) {
//        float[] keypoint = new float[2];
//        keypoint[0] = data[key + 2];
//        keypoint[1] = data[key + 3];
//        return keypoint;
//    }
//
//    public float[] iterator() {
//        return Arrays.copyOfRange(data, 2, data.length);
//    }
//
    public BBox getBbox() {
        float xmin = data[0];
        float ymin = data[1];
        float xmax = data[2];
        float ymax = data[3];
        return new BBox(xmin, ymin, xmax, ymax);
    }

    public float getScore() {
        return this.score;
    }

    public float[] getData() {
        return this.data;
    }

//
//    public Detection scaled(float factor) {
//        float[] scaledData = new float[data.length];
//        for (int i = 0; i < data.length; i++) {
//            scaledData[i] = data[i] * factor;
//        }
//        return new Detection(scaledData, score);
//    }


}

