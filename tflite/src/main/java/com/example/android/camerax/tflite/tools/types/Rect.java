package com.example.android.camerax.tflite.tools.types;

public class Rect {
    private float xCenter;
    private float yCenter;
    private float width;
    private float height;
    private float rotation;
    private boolean normalized;

    public Rect(float xCenter, float yCenter, float width, float height, float rotation, boolean normalized) {
        this.xCenter = xCenter;
        this.yCenter = yCenter;
        this.width = width;
        this.height = height;
        this.rotation = rotation;
        this.normalized = normalized;
    }

    public float getxCenter() {
        return xCenter;
    }

    public void setxCenter(float xCenter) {
        this.xCenter = xCenter;
    }

    public float getyCenter() {
        return yCenter;
    }

    public void setyCenter(float yCenter) {
        this.yCenter = yCenter;
    }

    public float getWidth() {
        return width;
    }

    public void setWidth(float width) {
        this.width = width;
    }

    public float getHeight() {
        return height;
    }

    public void setHeight(float height) {
        this.height = height;
    }

    public float getRotation() {
        return rotation;
    }

    public void setRotation(float rotation) {
        this.rotation = rotation;
    }

    public boolean isNormalized() {
        return normalized;
    }

    public void setNormalized(boolean normalized) {
        this.normalized = normalized;
    }

    public int[] getSize() {
        int[] size = new int[2];
        size[0] = (int)(this.width);
        size[1] = (int)(this.height);
        return size;
    }

    public float[] getSizeFloat() {
        float[] size = new float[2];
        size[0] = this.width;
        size[1] = this.height;
        return size;
    }


    public Rect scaled(float[] size, boolean normalize) {
        if (this.normalized == normalize) {
            return this;
        }
        float sx = size[0], sy = size[1];
        if (normalize) {
            sx = 1 / sx;
            sy = 1 / sy;
        }
        return new Rect(this.xCenter * sx, this.yCenter * sy,
                this.width * sx, this.height * sy,
                this.rotation, false);
    }

    public float[][] points() {
        float x = this.xCenter;
        float y = this.yCenter;
        float w = this.width / 2;
        float h = this.height / 2;
        float[][] pts = {{x - w, y - h}, {x + w, y - h}, {x + w, y + h}, {x - w, y + h}};
        if (this.rotation == 0) {
            return pts;
        }
        double s = Math.sin(this.rotation);
        double c = Math.cos(this.rotation);
        float[][] t = new float[4][2];
        for(int i = 0; i<4; i++) {
            t[i][0] = pts[i][0] - x;
            t[i][1] = pts[i][1] - y;
        }
        float[][] res = new float[4][2];
        for(int i = 0; i<4; i++) {
            res[i][0] = (float) (t[i][0] * c + t[i][1] * s);
            res[i][1] = (float) (-t[i][0] * s + t[i][1] * c);
        }
        for(int i = 0; i<4; i++) {
            res[i][0] += x;
            res[i][1] += y;
        }
        return res;
    }
}
