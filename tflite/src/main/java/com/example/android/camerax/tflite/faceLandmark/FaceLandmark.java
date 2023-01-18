package com.example.android.camerax.tflite.faceLandmark;

import com.example.android.camerax.tflite.tools.types.BBox;
import com.example.android.camerax.tflite.tools.types.Detection;
import com.example.android.camerax.tflite.tools.types.Rect;

public class FaceLandmark {
    float[] ROI_SCALE = {1.5f, 1.5f};
    public float[] _select_roi_size(BBox bbox, float[] image_size) {
        int  abs_width, abs_height;

        abs_width = (int) (bbox.width * image_size[0]);
        abs_height = (int) (bbox.height * image_size[1]);

        int width = abs_width;
        int height = abs_height;
        int long_size = Math.max(width, height);
        float[] sizes = new float[2];
        sizes[0] = (float) long_size / image_size[0];
        sizes[1] = (float) long_size / image_size[1];
        return sizes;
    }

    public Rect face_detection_to_roi(Detection face_detection, float[] image_size){

        return bbox_to_roi(face_detection.getBbox(), image_size, ROI_SCALE);
    }
    public Rect bbox_to_roi(BBox bbox, float[] image_size, float[] scale) {

        // select ROI dimensions
        float[] size = _select_roi_size(bbox, image_size);
        //added adinionally
        bbox = bbox.scale(image_size);
        //
        float width = size[0] * scale[0];
        float height = size[1] * scale[1];
        // calculate ROI size and -centre
        float cx = bbox.xmin + bbox.width / 2;
        float cy = bbox.ymin + bbox.height / 2;
        // calculate rotation of required
        return new Rect(cx, cy, width, height, 0.0f,false);
    }


}
