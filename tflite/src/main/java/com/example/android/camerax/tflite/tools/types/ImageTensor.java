package com.example.android.camerax.tflite.tools.types;

import java.util.Arrays;

public class ImageTensor {
    private float[] tensorData;
    // Padding values (left, right, top, bottom)
    private float[] padding;
    private int[] originalSize;

    public ImageTensor(float[] tensorData, float[] padding, int[] originalSize) {
        this.tensorData = tensorData;
        this.padding = padding;
        this.originalSize = originalSize;
    }

    public float[] getTensorData() {
        return tensorData;
    }

    public void setTensorData(float[] tensorData) {
        this.tensorData = tensorData;
    }

    public float[] getPadding() {
        return padding;
    }

    public void setPadding(float[] padding) {
        this.padding = padding;
    }

    public int[] getOriginalSize() {
        return originalSize;
    }

    public void setOriginalSize(int[] originalSize) {
        this.originalSize = originalSize;
    }

    @Override
    public String toString() {
        return "ImageTensor{" +
                "tensorData=" + Arrays.toString(tensorData) +
                ", padding=" + Arrays.toString(padding) +
                ", originalSize=" + Arrays.toString(originalSize) +
                '}';
    }
}
