#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

const int LED_PIN = LED_BUILTIN;
const int SERVO_CHANNEL = 0;

const int front_left_shoulder = 8; //--> min endevant, max (2750)--> enrera
const int front_left_elbow = 9; //--> min endevant, max (2750) --> enrera

const int back_left_shoulder = 10; // min: endevant, max  (2500): enderrera
const int back_left_elbow = 11; // min: endevant, max  (2750): enderrera

const int front_right_shoulder = 12; //--> min enrera, max (2500)--> endevant
const int front_right_elbow = 13; //--> min enrera, max (2750) --> endevant

const int back_right_shoulder = 14; //--> min enrera, max (2500)--> endevant
const int back_right_elbow = 15; //--> min enrera, max (2750) --> endevant


// Valors prudents inicials.
// Molts servos accepten aproximadament 1000-2000 us.
// Alguns arriben a 500-2500 us, però no comencis pels extrems.
const int SERVO_MIN_US = 500;  
const int SERVO_MAX_US = 2500; // 995 1500 --> 180º
const int SERVO_MAX2_US = 2750; //MG90s and 996 2750 --> 180º
const int SERVO_HALF_US = 1500; 
const int SERVO_HALF2_US = 1625; 


void setup() {
  pinMode(LED_PIN, OUTPUT);

  Wire.begin();

  pca.begin();
  pca.setPWMFreq(50);  // Servos típics: 50 Hz

  delay(10);
}


void fold() {
  
  pca.writeMicroseconds(back_left_elbow, SERVO_MAX2_US);
  pca.writeMicroseconds(back_right_elbow, SERVO_MIN_US);
  delay(2000);
  pca.writeMicroseconds(front_left_elbow, SERVO_MIN_US);
  pca.writeMicroseconds(front_right_elbow, SERVO_MAX2_US);
  delay(2000);
  pca.writeMicroseconds(back_left_shoulder, SERVO_MIN_US);
  pca.writeMicroseconds(back_right_shoulder, SERVO_MAX_US);
  delay(2000);
  pca.writeMicroseconds(front_left_shoulder, SERVO_MAX2_US);
  pca.writeMicroseconds(front_right_shoulder, SERVO_MIN_US);
  delay(2000);
}


void get_up() {
  pca.writeMicroseconds(front_left_shoulder, SERVO_HALF_US);
  pca.writeMicroseconds(front_right_shoulder, SERVO_HALF_US);
  delay(2000);
  pca.writeMicroseconds(back_left_shoulder, SERVO_HALF_US);
  pca.writeMicroseconds(back_right_shoulder, SERVO_HALF_US);
  delay(2000);
  pca.writeMicroseconds(front_left_elbow, SERVO_MAX2_US);
  pca.writeMicroseconds(front_right_elbow, SERVO_MIN_US);
  delay(2000);
  pca.writeMicroseconds(back_left_elbow, SERVO_MIN_US);
  pca.writeMicroseconds(back_right_elbow, SERVO_MAX2_US);
  delay(2000);
}


void loop() {
  // Posició fold
  fold();
  // pca.writeMicroseconds(back_right_elbow, SERVO_MIN_US);
  delay(5000);
  //pca.writeMicroseconds(back_right_elbow, SERVO_MAX2_US);
  //get_up();
  delay(5000);
}