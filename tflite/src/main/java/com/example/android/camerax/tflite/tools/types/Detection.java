package com.example.android.camerax.tflite.tools.types;

public class Detection {
    private float[] data;
    private float score;

    public Detection(float[] data, float score) {
        this.data = data;
        this.score = score;
    }

}
