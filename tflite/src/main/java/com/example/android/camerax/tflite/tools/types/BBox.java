package com.example.android.camerax.tflite.tools.types;

public class BBox {
    public float xmin;
    public float ymin;
    public float xmax;
    public float ymax;
    public float width;
    public float height;

    public BBox(float xmin, float ymin, float xmax, float ymax) {
        this.xmin = xmin;
        this.ymin = ymin;
        this.xmax = xmax;
        this.ymax = ymax;
        this.width = xmax - xmin;
        this.height = ymax - ymin;
    }
    // never used
    // public float[] getAsTuple() {
    //     float[] tuple = new float[4];
    //     tuple[0] = this.xmin;
    //     tuple[1] = this.ymin;
    //     tuple[2] = this.xmax;
    //     tuple[3] = this.ymax;
    //     return tuple;
    // }

    public float getWidth() {
        return this.xmax - this.xmin;
    }

    public float getHeight() {
        return this.ymax - this.ymin;
    }

    public boolean isEmpty() {
        return this.getWidth() <= 0 || this.getHeight() <= 0;
    }

    public boolean isNormalized() {
        return this.xmin >= -1 && this.xmax < 2 && this.ymin >= -1;
    }

    public float getArea() {
        if (this.isEmpty()) {
            return 0;
        }
        return this.getWidth() * this.getHeight();
    }

    public BBox intersect(BBox other) {
        float xmin = Math.max(this.xmin, other.xmin);
        float ymin = Math.max(this.ymin, other.ymin);
        float xmax = Math.min(this.xmax, other.xmax);
        float ymax = Math.min(this.ymax, other.ymax);
        if (xmin < xmax && ymin < ymax) {
            return new BBox(xmin, ymin, xmax, ymax);
        }
        else {
            return null;
        }
    }

    public BBox scale(float[] size) {
        float sx = size[0];
        float sy = size[1];
        float xmin = this.xmin * sx;
        float ymin = this.ymin * sy;
        float xmax = this.xmax * sx;
        float ymax = this.ymax * sy;
        return new BBox(xmin, ymin, xmax, ymax);
    }

    public BBox absolute(float[] size) {
        if (!this.isNormalized()) {
            return this;
        }
        return this.scale(size);
    }
}
