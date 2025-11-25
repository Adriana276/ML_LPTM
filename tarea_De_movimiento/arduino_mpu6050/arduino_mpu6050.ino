/*
 * Arduino/ESP32 MPU6050 Data Acquisition for Real-Time HAR
 * Sends accelerometer and gyroscope data via Serial
 * Compatible with Arduino Uno and ESP32
 */

#include <Wire.h>

// MPU6050 I2C address
const int MPU_ADDR = 0x68;

// Data variables
int16_t acc_x, acc_y, acc_z;
int16_t gyro_x, gyro_y, gyro_z;
int16_t temp;

// Timing
unsigned long lastSample = 0;
const int SAMPLE_RATE = 50;                     // Hz
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE; // ms

void setup() {
  Serial.begin(115200);

  // Initialize I2C
  Wire.begin();

  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Wake up
  Wire.endTransmission(true);

  // Configure accelerometer (±2g)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); // ACCEL_CONFIG register
  Wire.write(0x00); // ±2g
  Wire.endTransmission(true);

  // Configure gyroscope (±250°/s)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); // GYRO_CONFIG register
  Wire.write(0x00); // ±250°/s
  Wire.endTransmission(true);

  delay(100);

  Serial.println("MPU6050 initialized");
  Serial.println(
      "Sending data format: !acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z@");
  delay(1000);
}

void loop() {
  unsigned long currentTime = millis();

  // Sample at fixed rate
  if (currentTime - lastSample >= SAMPLE_INTERVAL) {
    lastSample = currentTime;

    // Read accelerometer and gyroscope data
    readMPU6050();

    // Convert to g and deg/s
    float ax = acc_x / 16384.0; // ±2g range
    float ay = acc_y / 16384.0;
    float az = acc_z / 16384.0;
    float gx = gyro_x / 131.0; // ±250°/s range
    float gy = gyro_y / 131.0;
    float gz = gyro_z / 131.0;

    // Send data in format: !ax,ay,az,gx,gy,gz@
    Serial.print("!");
    Serial.print(ax, 3);
    Serial.print(",");
    Serial.print(ay, 3);
    Serial.print(",");
    Serial.print(az, 3);
    Serial.print(",");
    Serial.print(gx, 2);
    Serial.print(",");
    Serial.print(gy, 2);
    Serial.print(",");
    Serial.print(gz, 2);
    Serial.println("@");
  }
}

void readMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Starting register for accelerometer
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true); // Request 14 bytes

  // Read accelerometer
  acc_x = Wire.read() << 8 | Wire.read();
  acc_y = Wire.read() << 8 | Wire.read();
  acc_z = Wire.read() << 8 | Wire.read();

  // Temperature (not used)
  temp = Wire.read() << 8 | Wire.read();

  // Read gyroscope
  gyro_x = Wire.read() << 8 | Wire.read();
  gyro_y = Wire.read() << 8 | Wire.read();
  gyro_z = Wire.read() << 8 | Wire.read();
}
