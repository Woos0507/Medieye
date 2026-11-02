#include "esp_camera.h"
#include <Adafruit_NeoPixel.h>
#include <HTTPClient.h>
#include <WiFi.h>

// Set these for the makerthon network. Do not commit real credentials.
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *SERVER_URL = "http://192.168.0.10:8000/scan";
const char *TARGET_INFO = "expiry";

// XIAO ESP32-S3 Sense camera pin map (OV2640/OV3660 carrier).
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 10
#define CAM_PIN_SIOD 40
#define CAM_PIN_SIOC 39
#define CAM_PIN_Y9 48
#define CAM_PIN_Y8 11
#define CAM_PIN_Y7 12
#define CAM_PIN_Y6 14
#define CAM_PIN_Y5 16
#define CAM_PIN_Y4 18
#define CAM_PIN_Y3 17
#define CAM_PIN_Y2 15
#define CAM_PIN_VSYNC 38
#define CAM_PIN_HREF 47
#define CAM_PIN_PCLK 13

constexpr int BUTTON_SCAN = 1;
constexpr int BUTTON_ESSENTIAL = 2;
constexpr int BUTTON_EXTRA = 3;
constexpr int PIXEL_PIN = 4;
constexpr int PIXEL_COUNT = 24;

Adafruit_NeoPixel ring(PIXEL_COUNT, PIXEL_PIN, NEO_GRB + NEO_KHZ800);
bool cameraReady = false;
unsigned long lastButtonMs = 0;

bool pressed(int pin) {
  return digitalRead(pin) == LOW;
}

void setRing(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < PIXEL_COUNT; i++) ring.setPixelColor(i, ring.Color(r, g, b));
  ring.show();
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAM_PIN_Y2;
  config.pin_d1 = CAM_PIN_Y3;
  config.pin_d2 = CAM_PIN_Y4;
  config.pin_d3 = CAM_PIN_Y5;
  config.pin_d4 = CAM_PIN_Y6;
  config.pin_d5 = CAM_PIN_Y7;
  config.pin_d6 = CAM_PIN_Y8;
  config.pin_d7 = CAM_PIN_Y9;
  config.pin_xclk = CAM_PIN_XCLK;
  config.pin_pclk = CAM_PIN_PCLK;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  return esp_camera_init(&config) == ESP_OK;
}

void scanAndSend() {
  if (!cameraReady) {
    Serial.println("Camera is not ready.");
    setRing(255, 0, 0);
    return;
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi is not connected.");
    setRing(255, 0, 0);
    return;
  }

  setRing(0, 0, 180); // blue: capturing/sending
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed.");
    setRing(255, 0, 0);
