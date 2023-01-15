package com.example.android.camerax.tflite.faceDetection;

import java.util.ArrayList;
import java.util.List;

public final class SSD {
    public static final int numLayers = 4;
    public static final int inputSizeHeight = 128;
    public static final int inputSizeWidth = 128;
    public static final float anchorOffsetX = 0.5f;
    public static final float anchorOffsetY = 0.5f;
    public static final int[] strides = {8, 16, 16, 16};
    public static final float interpolatedScaleAspectRatio = 1.0f;

    public static float[][] ssdGenerateAnchors() {
        List<float[]> anchors = new ArrayList<>();
        int layerId = 0;
        int numLayers = SSD.numLayers;
        int[] strides = SSD.strides;
        assert strides.length == numLayers;
        float anchorOffsetX = SSD.anchorOffsetX;
        float anchorOffsetY = SSD.anchorOffsetY;
        float interpolatedScaleAspectRatio = SSD.interpolatedScaleAspectRatio;
        while (layerId < numLayers) {
            int lastSameStrideLayer = layerId;
            int repeats = 0;
            while (lastSameStrideLayer < numLayers && strides[lastSameStrideLayer] == strides[layerId]) {
                lastSameStrideLayer++;
                repeats += 2;
            }
            int stride = strides[layerId];
            int featureMapHeight = SSD.inputSizeHeight / stride;
            int featureMapWidth = SSD.inputSizeWidth / stride;
            for (int y = 0; y < featureMapHeight; y++) {
                float yCenter = (y + anchorOffsetY) / featureMapHeight;
                for (int x = 0; x < featureMapWidth; x++) {
                    float xCenter = (x + anchorOffsetX) / featureMapWidth;
                    for (int i = 0; i < repeats; i++) {
                        anchors.add(new float[]{xCenter, yCenter});
                    }
                }
            }
            layerId = lastSameStrideLayer;
        }
        float[][] anchorArray = new float[anchors.size()][2];
        for (int i = 0; i < anchors.size(); i++) {
            anchorArray[i] = anchors.get(i);
        }
        return anchorArray;
    }

}
