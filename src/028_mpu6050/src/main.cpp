// Streams raw MPU6050 data to the PC as CSV, one line per sample:
//   micros,ax,ay,az,gx,gy,gz,temp
// Values are already in physical units (m/s^2, rad/s, degC) because the
// Adafruit library scales them; a simple Python reader is
//   line.split(",") -> [float(x) for x in parts].

#include <Adafruit_AHRS.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;
Adafruit_Mahony filter;

void setup(void) {
  Serial.begin(230400);
  while (!mpu.begin()) {
    Serial.println("MPU not yet ready");
    delay(10);
  }
  Wire.setWireTimeout(3000, true);
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_184_HZ);
  filter.begin(100);
}

void loop(void) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  filter.updateIMU(g.gyro.x * 57.2958f, g.gyro.y * 57.2958f,
                   g.gyro.z * 57.2958f, a.acceleration.x / 9.80665f,
                   a.acceleration.y / 9.80665f, a.acceleration.z / 9.80665f);

  Serial.print(filter.getRoll());
  Serial.print(",");
  Serial.print(filter.getPitch());
  Serial.print(",");
  Serial.println(filter.getYaw());
}
