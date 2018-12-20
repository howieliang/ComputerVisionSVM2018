//*********************************************
// Computer Vision for Classification
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import gab.opencv.*;
import processing.video.*;
import java.awt.*;

Capture video;
OpenCV opencv;

PImage img, imgGray, imgR, imgG, imgB, imgH, imgS, imgV;
Histogram grayHist; 
Histogram rHist, gHist, bHist;
Histogram hHist, sHist, vHist;

int div = 2, binDiv = 1;
int w = 330, h = 250, d = 10;

void setup() {
  size(1000, 760);
  video = new Capture(this, 640/div, 480/div);
  opencv = new OpenCV(this, 640/div, 480/div);

  video.start();
  imageMode(CENTER);
}

void draw() {
  background(0);
  binDiv = (int)map(mouseX, 0, width, 0, 6);

  opencv.useGray();
  opencv.loadImage(video);
  imgGray = opencv.getSnapshot();
  grayHist = opencv.findHistogram(opencv.getGray(), (int)(256/pow(2, binDiv)));
  
  opencv.useColor();
  opencv.loadImage(video);
  rHist = opencv.findHistogram(opencv.getR(), (int)(256/pow(2, binDiv)));
  gHist = opencv.findHistogram(opencv.getG(), (int)(256/pow(2, binDiv)));
  bHist = opencv.findHistogram(opencv.getB(), (int)(256/pow(2, binDiv)));
  imgR = opencv.getSnapshot(opencv.getR());
  imgG = opencv.getSnapshot(opencv.getG());
  imgB = opencv.getSnapshot(opencv.getB()); 

  img = opencv.getSnapshot();
  image(img, d+video.width/2, d+video.height/2, video.width, video.height);

  opencv.useColor(HSB);
  opencv.loadImage(video);
  hHist = opencv.findHistogram(opencv.getH(), (int)(256/pow(2, binDiv)));
  sHist = opencv.findHistogram(opencv.getS(), (int)(256/pow(2, binDiv)));  
  vHist = opencv.findHistogram(opencv.getV(), (int)(256/pow(2, binDiv)));
  imgH = opencv.getSnapshot(opencv.getH());
  imgS = opencv.getSnapshot(opencv.getS());  
  imgV = opencv.getSnapshot(opencv.getV());
  
  drawImage(imgGray, grayHist, d+ 2*w, d + 0*h, w-d, h-d, color(255));
  drawImage(imgR, rHist, d+ 0*w, d + 1*h, w-d, h-d, color(255, 0, 0));
  drawImage(imgG, gHist, d+ 1*w, d + 1*h, w-d, h-d, color(0, 255, 0));
  drawImage(imgB, bHist, d+ 2*w, d + 1*h, w-d, h-d, color(0, 0, 255));
  drawImage(imgH, hHist, d+ 0*w, d + 2*h, w-d, h-d, color(255));
  drawImage(imgS, sHist, d+ 1*w, d + 2*h, w-d, h-d, color(255));
  drawImage(imgV, vHist, d+ 2*w, d + 2*h, w-d, h-d, color(255));
}

void captureEvent(Capture c) {
  c.read();
}

void drawImage(PImage imgR, Histogram rHist, int x, int y, int w, int h, color c) {
  stroke(c); 
  noFill();
  pushStyle();
  imageMode(CORNER);
  tint(c);
  image(imgR, x, y, w, h);
  popStyle();
  rHist.draw(x, y, w, h);
  stroke(c); 
  noFill();  
  rect(x, y, w, h);
}