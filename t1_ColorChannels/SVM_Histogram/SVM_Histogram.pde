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

Histogram grayHist; 
Histogram rHist, gHist, bHist;
Histogram hHist, sHist, vHist;
PImage img, imgGray, imgR, imgG, imgB, imgH, imgS, imgV;

int div = 2, binDiv = 5;
int w = 330, h = 250, dt = 10;

int sensorNum = 8; //number of sensors in use
int dataNum = 500; //number of data to show
float[] rawData = new float[sensorNum];
float[][] sensorHist = new float[sensorNum][dataNum]; //history data to show
float[] modeArray = new float[dataNum]; //classification to show
//SVM parameters
double C = 64; //Cost: The regularization parameter of SVM
int d = sensorNum;     //Number of features to feed into the SVM

void setup() {
  size(1000, 760);
  video = new Capture(this, 640/div, 480/div);
  opencv = new OpenCV(this, 640/div, 480/div);

  video.start();
  imageMode(CENTER);

  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }
}

void draw() {
  background(0);

  if (!svmTrained && firstTrained) {
    //train a linear support vector classifier (SVC) 
    trainLinearSVC(d, C);
  }

  opencv.useGray();
  opencv.loadImage(video);
  imgGray = opencv.getSnapshot();
  grayHist = opencv.findHistogram(opencv.getGray(), (int)(256/pow(2, binDiv)));
  Mat matGray = grayHist.getMat();
  int numBins = matGray.height();
  for (int i = 0; i < numBins; i++) {
    appendArray(sensorHist[i], (float)matGray.get(i, 0)[0]);
    rawData[i] = (float)matGray.get(i, 0)[0];
  }

  img = opencv.getSnapshot();
  image(img, dt+video.width/2, dt+video.height/2, video.width, video.height);

  drawImage(imgGray, grayHist, dt+ 2*w, dt + 0*h, w-dt, h-dt, color(255));

  //Draw the sensor data
  for (int i = 0; i < sensorNum; i++) {
    lineGraph(sensorHist[i], 0, 1, 0, dt + 1*h+ i*height*0.05, width, height*0.05, i);
  }

  //Draw the modeArray
  //barGraph(float[] data, float lowerbound, float upperbound, float x, float y, float width, float height)
  barGraph(modeArray, 0, 100, 0, height, width, .1*height);

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
    }
  }
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
}

//Tool functions

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