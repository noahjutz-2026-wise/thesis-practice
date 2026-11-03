// Streams raw MPU6050 data to the PC as CSV, one line per sample:
//   micros,ax,ay,az,gx,gy,gz,temp
// Values are already in physical units (m/s^2, rad/s, degC) because the
// Adafruit library scales them; a simple Python reader is
//   line.split(",") -> [float(x) for x in parts].

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

void setup(void) {
  Serial.begin(230400);
  if (!mpu.begin()) {
    while (1)
      delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G); // 4096 LSB/g
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);      // 65.5 LSB/(deg/s)
  mpu.setFilterBandwidth(MPU6050_BAND_184_HZ);  // low lag, ok at ~500 Hz
}

void loop(void) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print(micros());
  Serial.print(',');
  Serial.print(a.acceleration.x, 4);
  Serial.print(',');
  Serial.print(a.acceleration.y, 4);
  Serial.print(',');
  Serial.print(a.acceleration.z, 4);
  Serial.print(',');
  Serial.print(g.gyro.x, 4);
  Serial.print(',');
  Serial.print(g.gyro.y, 4);
  Serial.print(',');
  Serial.print(g.gyro.z, 4);
  Serial.print(',');
  Serial.println(temp.temperature, 2);
}
