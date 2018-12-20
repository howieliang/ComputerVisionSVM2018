//*********************************************
// Computer Vision for Classification
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import gab.opencv.*;
import processing.video.*;
import java.awt.*;

OpenCV opencv;
Capture video;
PImage img, imgR, imgG, imgB, imgH, imgS, imgV;

int imgHeight, imgW;
int div = 2;

void setup() {
  size(960, 720);
  video = new Capture(this, 640/div, 480/div);
  opencv = new OpenCV(this, 640/div, 480/div);
  video.start();
}

void update() {
  opencv.loadImage(video);
  opencv.useColor();
  img = opencv.getSnapshot();
  imgHeight = img.height;
  imgW = img.width;

  imgR = opencv.getSnapshot(opencv.getR());
  imgG = opencv.getSnapshot(opencv.getG());
  imgB = opencv.getSnapshot(opencv.getB());  

  opencv.useColor(HSB);

  imgH = opencv.getSnapshot(opencv.getH());
  imgS = opencv.getSnapshot(opencv.getS());  
  imgV = opencv.getSnapshot(opencv.getV());
}

void draw() {
  background(0);
  update();
  
  noTint();
  image(img, 0, 0, imgW, imgHeight);

  tint(255, 0, 0);
  image(imgR, 0, imgHeight, imgW, imgHeight);

  tint(0, 255, 0);
  image(imgG, imgW, imgHeight, imgW, imgHeight);

  tint(0, 0, 255);
  image(imgB, 2*imgW, imgHeight, imgW, imgHeight);

  noTint();
  image(imgH, 0, 2*imgHeight, imgW, imgHeight);
  image(imgS, imgW, 2*imgHeight, imgW, imgHeight);
  image(imgV, 2*imgW, 2*imgHeight, imgW, imgHeight);
}

void captureEvent(Capture c) {
  c.read();
}