#include <Arduino.h>
#include "esp_camera.h"

#if !ARDUINO_USB_CDC_ON_BOOT || ARDUINO_USB_MODE != 1
#error "Select Hardware CDC and JTAG and enable USB CDC On Boot."
#endif

namespace {
constexpr size_t kMaxJpeg = 128 * 1024;
bool cameraReady = false;

bool initializeCamera() {
  if (!psramFound()) {
    Serial0.println("Enable OPI PSRAM for XIAO ESP32S3.");
    return false;
  }

  // XIAO ESP32S3 Sense expansion-board camera, from Seeed's pinout.
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 15;
  config.pin_d1 = 17;
  config.pin_d2 = 18;
  config.pin_d3 = 16;
  config.pin_d4 = 14;
  config.pin_d5 = 12;
  config.pin_d6 = 11;
  config.pin_d7 = 48;
  config.pin_xclk = 10;
  config.pin_pclk = 13;
  config.pin_vsync = 38;
  config.pin_href = 47;
  config.pin_sccb_sda = 40;
  config.pin_sccb_scl = 39;
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 14;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  const esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    Serial0.printf("Camera initialization failed (0x%x)\n", static_cast<unsigned>(result));
    return false;
  }
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    Serial0.printf("Camera sensor PID: 0x%x\n", sensor->id.PID);
    if (sensor->id.PID == OV3660_PID) {
      sensor->set_vflip(sensor, 1);
    }
  }
  return true;
}

bool writeAll(const uint8_t *data, size_t size) {
  while (size > 0) {
    size_t written = Serial.write(data, min(size, static_cast<size_t>(1024)));
    if (written == 0) {
      return false;
    }
    data += written;
    size -= written;
  }
  return true;
}

void sendImage(const camera_fb_t &frame) {
  if (frame.format != PIXFORMAT_JPEG || frame.len < 4 || frame.len > kMaxJpeg) {
    return;
  }
  const uint32_t size = frame.len;
  const uint8_t header[] = {
      'X', 'I', 'A', 'O',
      static_cast<uint8_t>(size), static_cast<uint8_t>(size >> 8),
      static_cast<uint8_t>(size >> 16), static_cast<uint8_t>(size >> 24)
  };
  if (writeAll(header, sizeof(header))) {
    writeAll(frame.buf, frame.len);
  }
}
}  // namespace

void setup() {
  Serial0.begin(115200);
  Serial.begin(115200);
  Serial.setTxTimeoutMs(100);
  cameraReady = initializeCamera();
}

void loop() {
  if (!cameraReady || !Serial) {
    delay(20);
    return;
  }
  camera_fb_t *frame = esp_camera_fb_get();
  if (frame == nullptr) {
    delay(20);
    return;
  }
  sendImage(*frame);
  esp_camera_fb_return(frame);
  delay(1);
}
