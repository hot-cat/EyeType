package com.example.android.camerax.tflite.tools.types;

import java.util.Arrays;

public class Detection {
    private float[][] data;
    private float score;

    public Detection(float[][] data, float score) {
        this.data = data;
        this.score = score;
    }

    public int getLength() {
        return data.length - 2;
    }

    public float[] getKeypoint(int key) {
        float[] keypoint = new float[2];
        keypoint[0] = data[key + 2][0];
        keypoint[1] = data[key + 2][1];
        return keypoint;
    }

    public float[][] getKeypoints() {
        return Arrays.copyOfRange(data, 2, data.length);
    }

    public BBox getBbox() {
        float xmin = data[0][0];
        float ymin = data[0][1];
        float xmax = data[1][0];
        float ymax = data[1][1];
        return new BBox(xmin, ymin, xmax, ymax);
    }

    public Detection scaled(float factor) {
        for (int i = 0; i < data.length; i++) {
            data[i][0] *= factor;
            data[i][1] *= factor;
        }
        return new Detection(data, score);
    }
}
