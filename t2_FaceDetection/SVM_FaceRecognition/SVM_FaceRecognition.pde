//*********************************************
// Computer Vision for Classification
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import gab.opencv.*;
import processing.video.*;
import java.awt.*;
import org.opencv.core.Mat;

Capture video;
OpenCV opencv;

PImage photo;
int grid_r = 640;
int grid_h = 480;

Histogram grayHist; 
Histogram rHist, gHist, bHist;
Histogram hHist, sHist, vHist;
PImage img, imgGray, imgR, imgG, imgB, imgH, imgS, imgV;

int div = 2, binDiv = 5;
int w = 330, h = 250, dt = 10;

int imgW = 10;
int sensorNum = imgW*imgW; //number of sensors in use
int dataNum = 100; //number of data to show
float[] rawData = new float[sensorNum];
float[][] sensorHist = new float[sensorNum][dataNum]; //history data to show
float[] modeArray = new float[dataNum]; //classification to show
//SVM parameters
double C = 64; //Cost: The regularization parameter of SVM
int d = sensorNum;     //Number of features to feed into the SVM
int lastPredY = -1;

void setup() {
  size(940, 480);
  video = new Capture(this, 640/div, 480/div);
  opencv = new OpenCV(this, 640/div, 480/div);
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);  
  video.start();
  imageMode(CENTER);

  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }
  
  noFill();
  stroke(0, 255, 0);
  strokeWeight(3);
}

void draw() {
  background(52);

  if (!svmTrained && firstTrained) {
    //train a linear support vector classifier (SVC) 
    trainLinearSVC(d, C);
  }

  opencv.loadImage(video);
  opencv.useColor();
  photo = opencv.getSnapshot();
  pushMatrix();
  scale(-div, div);
  translate(-640/(2*div), 480/(2*div));
  image(photo, 0, 0);
  popMatrix();

  Rectangle[] faces = opencv.detect();

  PImage[] faceImgs = new PImage[faces.length]; 
  PImage[] faceImgsGray = new PImage[faces.length]; 

  if (faces.length>0) {
    for (int i = 0; i < 1; i++) {
      faceImgs[i] = new PImage();
      faceImgsGray[i] = new PImage();
      faceImgs[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      faceImgs[i].resize(100, 100);
      faceImgsGray[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height); 
      faceImgsGray[i].filter(GRAY);
      faceImgsGray[i].resize(10, 10);

      pushStyle();
      strokeWeight(3);
      noFill();
      if(!svmTrained){
        stroke(255);
      }else{
        if(lastPredY>=0 && lastPredY<=9)stroke(colors[lastPredY]);
        else stroke(255);
      }

      pushMatrix();
      scale(-div, div);
      translate(-(640/div)+(faces[i].x), faces[i].y);
      println(i, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      rect(0, 0, faces[i].width, faces[i].height);
      popMatrix();
      popStyle();
      
      pushMatrix();
      translate(640 + 50, i*100+50);
      image(faceImgs[i], 0, 0);
      popMatrix();

      pushMatrix();
      translate(740 + 50, i*100+50);
      scale(10);
      image(faceImgsGray[i], 0, 0);
      popMatrix();

      pushMatrix();
      translate(840, i*100);
      pushStyle();
      noStroke();
      for (int x = 0; x < faceImgsGray[i].width; x++) {
        for (int y = 0; y < faceImgsGray[i].width; y++) {
          color c = faceImgsGray[i].get(x, y);
          fill(c);
          rect(x*10, y*10, 10, 10);
        }
      }
      popStyle();
      popMatrix();
    }

    for (int x = 0; x < faceImgsGray[0].width; x++) {
      for (int y = 0; y < faceImgsGray[0].width; y++) {
        int index = x + y * 10;
        color c = faceImgsGray[0].get(x, y) & 0xFF;
        appendArray(sensorHist[index], c);
        rawData[index] = c;
      }
    }

    //Draw the sensor data
    pushStyle();
    strokeWeight(1);
    stroke(255);

    barGraph(modeArray, 0, 100, 640, height, 300, .1*height);

    //lineGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h, int _index)
    for (int i = 0; i < sensorNum; i++) {
      lineGraph(sensorHist[i], 0, 255, 640, 100+(3.8)*i, 300, 5, 0);
    }

    popStyle();

    //use the data for classification
    double[] X = new double[d]; //Form a feature vector X;
    double[] dataToTrain = new double[d+1];
    double[] dataToTest = new double[d];
    if (mousePressed) {
      if (!svmTrained) { //if the SVM model is not trained
        int Y = type; //Form a label Y;
        for (int i = 0; i < d; i++) {
          X[i] = rawData[i];
          dataToTrain[i] = X[i];
        }
        dataToTrain[d] = Y;
        trainData.add(new Data(dataToTrain)); //Add the dataToTrain to the trainingData collection.
        appendArray(modeArray, Y); //append the label to  for visualization
        ++tCnt;
      } else { //if the SVM model is trained
        for (int i = 0; i < d; i++) {
          X[i] = rawData[i];
          dataToTest[i] = X[i];
        }
        int predictedY = (int) svmPredict(dataToTest); //SVMPredict the label of the dataToTest
        appendArray(modeArray, predictedY); //append the prediction results to modeArray for visualization
      }
    } else {
      if (!svmTrained) { //if the SVM model is not trained
        appendArray(modeArray, -1); //the class is null without mouse pressed.
      } else { //if the SVM model is trained
        for (int i = 0; i < d; i++) {
          X[i] = rawData[i];
          dataToTest[i] = X[i];
        }
        int predictedY = (int) svmPredict(dataToTest); //SVMPredict the label of the dataToTest
        appendArray(modeArray, predictedY); //append the prediction results to modeArray for visualization
        lastPredY = predictedY;
      }
    }
  }
}

void captureEvent(Capture c) {
  c.read();
}

void keyPressed() {
  if (key == ENTER) {
    if (tCnt>0 || type>0) {
      if (!firstTrained) firstTrained = true;
      resetSVM();
    } else {
      println("Error: No Data");
    }
  }
  if (key >= '0' && key <= '9') {
    type = key - '0';
  }
  if (key == TAB) {
    if (tCnt>0) { 
      if (type<(colors.length-1))++type;
      tCnt = 0;
    }
  }
  if (key == '/') {
    firstTrained = false;
    resetSVM();
    clearSVM();
  }
  if (key == 'S' || key == 's') {
    if (model!=null) { 
      saveSVM_Model(sketchPath()+"/data/test.model", model);
      println("Model Saved");
    }
  }
  //if (key == ' ') {
  //  if (b_pause == true) b_pause = false;
  //  else b_pause = true;
  //}
}

//Append a value to a float[] array.
float[] appendArray (float[] _array, float _val) {
  float[] array = _array;
  float[] tempArray = new float[_array.length-1];
  arrayCopy(array, tempArray, tempArray.length);
  array[0] = _val;
  arrayCopy(tempArray, 0, array, 1, tempArray.length);
  return array;
}

//Draw a line graph to visualize the sensor stream
void lineGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h, int _index) {
  color colors[] = {
    color(255, 0, 0), color(0, 255, 0), color(0, 0, 255), color(255, 255, 0), color(0, 255, 255), 
    color(255, 0, 255), color(0)
  };
  int index = min(max(_index, 0), colors.length);
  pushStyle();
  float delta = _w/data.length;
  beginShape();
  noFill();
  stroke(255);
  for (float i : data) {
    float h = map(i, _l, _u, 0, _h);
    vertex(_x, _y+h);
    _x = _x + delta;
  }
  endShape();
  popStyle();
}

//Draw a bar graph to visualize the modeArray
void barGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), color(82, 179, 217), color(244, 208, 63), 
    color(242, 121, 53), color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), color(128, 52, 0)
  };
  pushStyle();
  noStroke();
  float delta = _w / data.length;
  for (int p = 0; p < data.length; p++) {
    float i = data[p];
    int cIndex = min((int) i, colors.length-1);
    if (i<0) fill(255, 100);
    else fill(colors[cIndex], 100);
    float h = map(_u, _l, _u, 0, _h);
    rect(_x, _y-h, delta, h);
    _x = _x + delta;
  }
  popStyle();
}