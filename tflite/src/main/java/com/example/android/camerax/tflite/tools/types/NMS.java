package com.example.android.camerax.tflite.tools.types;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

public class NMS {
//    @RequiresApi(api = Build.VERSION_CODES.N)
//    public static List<Detection> non_maximum_suppression(List<Detection> detections, float min_suppression_threshold, Optional<Float> min_score, boolean weighted) {
//        List<Detection> result = new ArrayList<>();
//        float[] scores = new float[detections.size()];
//        for (int i = 0; i < detections.size(); i++) {
//            scores[i] = detections.get(i).score;
//        }
//
//        List<Integer> indexed_scores = new ArrayList<>();
//        for (int i = 0; i < scores.length; i++) {
//            indexed_scores.add(i);
//        }
//
//        indexed_scores.sort((a, b) -> Float.compare(scores[b], scores[a]));
//
//        if (weighted) {
//            return _weighted_non_maximum_suppression(indexed_scores, detections, min_suppression_threshold, min_score.orElse(null));
//        } else {
//            return _non_maximum_suppression(indexed_scores, detections, min_suppression_threshold, min_score.orElse(null));
//        }
//    }
//
//    public static List<Detection> _non_maximum_suppression(List<Tuple<Integer, Float>> indexed_scores, List<Detection> detections, float min_suppression_threshold, Optional<Float> min_score) {
//        List<BBox> kept_boxes = new ArrayList<>();
//        List<Detection> outputs = new ArrayList<>();
//        for (Tuple<Integer, Float> tuple : indexed_scores) {
//            // exit loop if remaining scores are below threshold
//            if (min_score.isPresent() && tuple.getB() < min_score.get()) {
//                break;
//            }
//            Detection detection = detections.get(tuple.getA());
//            BBox bbox = detection.getBbox();
//            boolean suppressed = false;
//            for (BBox kept : kept_boxes) {
//                float similarity = _overlap_similarity(kept, bbox);
//                if (similarity > min_suppression_threshold) {
//                    suppressed = true;
//                    break;
//                }
//            }
//            if (!suppressed) {
//                outputs.add(detection);
//                kept_boxes.add(bbox);
//            }
//        }
//        return outputs;
//    }
//
//    private static List<Detection> _weighted_non_maximum_suppression(List<Integer> indexed_scores, List<Detection> detections, float min_suppression_threshold, Float min_score) {
//        // implementation for _weighted_non_maximum_suppression
//        return new ArrayList<>();
//    }

    public static List<Detection> non_maximum_suppression(List<Detection> detections, float min_suppression_threshold, float min_score, boolean weighted) {
        List<Float> scores = new ArrayList<>();
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < detections.size(); i++) {
            scores.add(detections.get(i).score);
            indexes.add(i);
        }
        // Sort the scores and indexes in descending order
        List<Integer> sortedIndexes = new ArrayList<>(indexes);
        Collections.sort(sortedIndexes, (i1, i2) -> -Float.compare(scores.get(i1), scores.get(i2)));
        if (weighted) {
            return _weighted_non_maximum_suppression(sortedIndexes,scores, detections, min_suppression_threshold, min_score);
        } else {
            return _non_maximum_suppression(sortedIndexes,scores, detections, min_suppression_threshold, min_score);
        }
    }
    public static List<Detection> _non_maximum_suppression(
            List<Integer> indexes, List<Float> scores, List<Detection> detections,
            float min_suppression_threshold, float min_score) {

        List<BBox> kept_boxes = new ArrayList<>();
        List<Detection> outputs = new ArrayList<>();

        for (int i = 0; i < indexes.size(); i++) {
            // exit loop if remaining scores are below threshold
            if (scores.get(i) < min_score) {
                break;
            }
            Detection detection = detections.get(indexes.get(i));
            BBox bbox = detection.getBbox();
            boolean suppressed = false;
            for (BBox kept : kept_boxes) {
                float similarity = _overlap_similarity(kept, bbox);
                if (similarity > min_suppression_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                outputs.add(detection);
                kept_boxes.add(bbox);
            }
        }
        return outputs;
    }


    public static List<Detection> _weighted_non_maximum_suppression(List<Integer> indexes, List<Float> scores, List<Detection> detections, float min_suppression_threshold, float min_score) {
        List<Detection> remaining_indexed_scores = new ArrayList<>();
        List<Detection> remaining = new ArrayList<>();
        List<Detection> candidates = new ArrayList<>();
        List<Detection> outputs = new ArrayList<>();

        for (int i = 0; i < indexes.size(); i++) {
            remaining_indexed_scores.add(new Detection(detections.get(indexes.get(i)).data, scores.get(i)));
        }

        while (!remaining_indexed_scores.isEmpty()) {
            Detection detection = remaining_indexed_scores.get(0);
            // exit loop if remaining scores are below threshold
            if (min_score != 0 && detection.score < min_score) {
                break;
            }
            int num_prev_indexed_scores = remaining_indexed_scores.size();
            BBox detection_bbox = detection.getBbox();
            remaining.clear();
            candidates.clear();
            Detection weighted_detection = detection;
            for (Detection d : remaining_indexed_scores) {
                BBox remaining_bbox = d.getBbox();
                float similarity = _overlap_similarity(remaining_bbox, detection_bbox);
                if (similarity > min_suppression_threshold) {
                    candidates.add(d);
                } else {
                    remaining.add(d);
                }
            }
            // weighted merging of similar (close) boxes
            if (!candidates.isEmpty()) {
                float[] weighted = new float[detection.data.length];
                float total_score = 0;
                for (Detection d : candidates) {
                    total_score += d.score;
                    for (int j = 0; j < weighted.length; j++) {
                        weighted[j] += d.data[j] * d.score;
                    }
                }
                for (int j = 0; j < weighted.length; j++) {
                    weighted[j] /= total_score;
                }
                weighted_detection = new Detection(weighted, detection.score);
            }
            outputs.add(weighted_detection);
            // exit the loop if the number of indexed scores didn't change
            if (num_prev_indexed_scores == remaining.size()) {
                break;
            }
            remaining_indexed_scores = new ArrayList<>(remaining);
        }
        return outputs;
    }

    public static float _overlap_similarity(BBox box1, BBox box2) {
        BBox intersection = box1.intersect(box2);
        if (intersection == null) {
            return 0.0f;
        }
        float intersect_area = intersection.getArea();
        float denominator = box1.getArea() + box2.getArea() - intersect_area;
        return (denominator > 0.0f) ? (intersect_area / denominator) : 0.0f;
    }


}