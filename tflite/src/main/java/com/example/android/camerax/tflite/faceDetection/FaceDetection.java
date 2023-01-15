package com.example.android.camerax.tflite.faceDetection;

import com.example.android.camerax.tflite.tools.types.Detection;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.util.ArrayList;
import java.util.List;

public class FaceDetection {

    //used in decode boxes
    float[][] anchors = SSD.ssdGenerateAnchors();

    public float[][][] decodeBoxes(float[][][] raw_boxes) {
        // width == height so scale is the same across the board
        // 128 got from face detection model input size
        float scale = 128;
        int num_points = raw_boxes[0][0].length / 2;
        // scale all values (applies to positions, width, and height alike)
        float[][][] boxes = new float[raw_boxes[0].length][num_points][2];
        for(int i = 0; i < raw_boxes[0].length; i++) {
            for(int j = 0; j < num_points; j++) {
                for(int k = 0; k < 2; k++) {
                    boxes[i][j][k] = raw_boxes[0][i][j*2+k]/scale;
                }
            }
        }
        // adjust center coordinates and key points to anchor positions
        for (int i = 0; i < boxes.length; i++) {
            boxes[i][0][0] += this.anchors[i][0];
            boxes[i][0][1] += this.anchors[i][1];
        }
        for (int i = 2; i < num_points; i++) {
            for (int j = 0; j < boxes.length; j++) {
                boxes[j][i][0] += this.anchors[j][0];
                boxes[j][i][1] += this.anchors[j][1];
            }
        }
        // convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        for(int i = 0; i < boxes.length; i++) {
            float[] center = new float[] {boxes[i][0][0], boxes[i][0][1]};
            float[] half_size = new float[] {boxes[i][1][0]/2, boxes[i][1][1]/2};
            boxes[i][0][0] = center[0] - half_size[0];
            boxes[i][0][1] = center[1] - half_size[1];
            boxes[i][1][0] = center[0] + half_size[0];
            boxes[i][1][1] = center[1] + half_size[1];
        }
        return boxes;
    }



    private static final float RAW_SCORE_LIMIT = 80.0f;
    public float[][][] getSigmoidScores(float[][][] raw_scores) {
            for (int i = 0; i < raw_scores.length; i++) {
                for (int j = 0; j < raw_scores[0].length; j++) {
                    for (int k = 0; k < raw_scores[0][0].length; k++) {
                        if (raw_scores[i][j][k] < -RAW_SCORE_LIMIT) {
                            raw_scores[i][j][k] = -RAW_SCORE_LIMIT;
                        } else if (raw_scores[i][j][k] > RAW_SCORE_LIMIT) {
                            raw_scores[i][j][k] = RAW_SCORE_LIMIT;
                        }
                        raw_scores[i][j][k] = 1.0f / (1.0f + (float) Math.exp(-raw_scores[i][j][k]));
                    }
                }
            }
            return raw_scores;
    }
    float MIN_SCORE = 0.5f; // you can set this to whatever threshold you want

//    public List<Detection> convertToDetections(float[][][] boxes, float[][][] scores) {
//        List<Detection> detections = new ArrayList<Detection>();
//        for (int i = 0; i < scores.length; i++) {
//            for (int j = 0; j < boxes[i].length; j++) {
//                if (scores[i][j][0] > MIN_SCORE && is_valid(boxes[i][j])) {
//                    detections.add(new Detection(boxes[i][j], scores[i][j][0]));
//                }
//            }
//        }
//        return detections;
//    }
//
//    private static boolean is_valid(float[] box) {
//        return box[1] > box[0];
//    }

    public List<Detection> convertToDetections(float[][][] boxes, float[][][] scores) {
        List<Detection> detections = new ArrayList<>();
        for (int i = 0; i < boxes.length; i++) {
            boolean valid = true;
            for (int j = 0; j < boxes[i].length; j++) {

                    if (boxes[i][j][1] <= boxes[i][j][0]) {
                        valid = false;
                        break;
                    }


            }
            if (valid && scores[0][i][0] > MIN_SCORE) {
                detections.add(new Detection(boxes[i], scores[0][i][0]));
            }
        }
        return detections;
    }

    public FaceDetection(){

    }
}
