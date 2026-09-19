/*
 * Layer 1 - ESP32 field node (padi_sawah)
 * ---------------------------------------
 * Sensing 17 kanal agro-meteo + upload 15 menit ke Laravel (POST /api/kirim_sensor).
 * Aktuasi mengikuti manuscript sec. 3.6 / Algorithm 1 step 15-16:
 *   relayPin  (GPIO19) -> UV lamp (lampu perangkap UV)
 *   relayPin2 (GPIO23) -> ultrasonic pest repeller
 *   relayPin3 (GPIO27) -> irrigation valve / notifikasi irigasi
 * Perintah aktuasi diterima dari model fusi CNN-LSTM lewat:
 *   (a) MQTT  : subscribe  taniverse/{iot_id}/inference        (real-time, utama)
 *   (b) REST  : GET https://petaniasik.my.id/api/inference/{id} (fallback tiap 15 dtk)
 * plus aturan lokal: BH1750 < 100 lx -> UV ON ; LiDAR gerakan -> ultrasonic ON.
 *
 * Library tambahan: PubSubClient (knolleary)  -- selain DHT, BH1750, Adafruit_BMP280,
 * WiFiManager, ArduinoJson yang sudah dipakai.
 */
#include "DHT.h"
#include <Wire.h>
#include <BH1750.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>   // Algorithm 1 step 16: subscribe perintah aktuasi via MQTT

//Link API
const char* serverName = "https://petaniasik.my.id/api/kirim_sensor";  // sesuaikan
const String iot_id = "jTZids5M";
const String token = "lOsx0KMJSAfJpxEe";
const String serverGet = "https://petaniasik.my.id/api/bacajson/" + iot_id;
// Hasil model fusi CNN-LSTM (Algorithm 1): keputusan aktuasi (UV / ultrasonik / irigasi)
const String serverInference = "https://petaniasik.my.id/api/inference/" + iot_id;

// ===== MQTT (broker aktuator, Algorithm 1 step 16) =====
const char* MQTT_HOST = "petaniasik.my.id";   // host broker MQTT
const int   MQTT_PORT = 1883;
const String MQTT_TOPIC = "taniverse/" + iot_id + "/inference";   // == config/mqtt.php topic_prefix
WiFiClient mqttNet;
PubSubClient mqtt(mqttNet);
unsigned long lastMqttReconnect = 0;
unsigned long lastAktuasiPoll = 0;

// ===== Status aktuator (Algorithm 1 step 15) =====
// action:  activate_ultrasonic_repeller_and_uv_lamp | recommend_selective_insecticide_spraying
//          irrigation_notification | continue_monitoring
bool cmd_uv = false;          // UV lamp    (relayPin)
bool cmd_ultrasonic = false;  // ultrasonic pest repeller (relayPin2)
bool cmd_irrigation = false;  // irrigation valve/alert   (relayPin3)
bool local_uv_night = false;      // BH1750: lampu perangkap UV saat malam
bool local_ultrasonic_lidar = false;  // LiDAR: gerakan terdeteksi -> pengusir ultrasonik

// Relay aktif-LOW (samakan dgn kontrol(): LOW = ON)
#define RELAY_ON  LOW
#define RELAY_OFF HIGH

// RS485
// Komunikasi via UART2 (TX=17, RX=16)
#define SerialPort Serial2
// Modbus RTU Query untuk masing-masing parameter
uint8_t npkQuery[]             = {0x01, 0x03, 0x00, 0x1E, 0x00, 0x03, 0x65, 0xCD};
uint8_t phQuery[]              = {0x01, 0x03, 0x00, 0x06, 0x00, 0x01, 0x64, 0x0B};
uint8_t soilMoistureQuery[]    = {0x01, 0x03, 0x00, 0x12, 0x00, 0x01, 0x24, 0x0F};
uint8_t soilTemperatureQuery[] = {0x01, 0x03, 0x00, 0x13, 0x00, 0x01, 0x75, 0xCF};

uint16_t nitrogen, phosphorus, potassium, phRaw, moistureRaw, tempRaw;
float nitrogen2 =0;
float phosphorus2 =0;
float potassium2 =0;
float pH = 0; 
float moisture = 0;
float temperature =0; 
int baterai = 0;

// Pin kontrol DE/RE
#define DE_RE_PIN 25
#define sekat 26

// Sinyal
long rssi;
int quality =0;

// Baterai
const int adcPin = 34; // Gunakan GPIO34 sebagai input ADC
const float R1 = 82000.0; // Ohm
const float R2 = 10000.0;  // Ohm
const float maxADC = 4095.0; // Resolusi ADC 12-bit
const float vRef = 3.3; // Tegangan referensi maksimum ADC

// DHT 22 (Suhu dan Kelembapan Udara)
#define DHTPIN 15
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);
float kelembapan_udara = 0;
float suhu_udara = 0;

// anemometer parameters (Kecepatan Angin)
volatile byte rpmcount;  // hitung signals
volatile unsigned long last_micros;
unsigned long timeold;
unsigned long timemeasure = 10.00;  // detik
int timetoSleep = 1;                // menit
unsigned long sleepTime = 15;       // menit
unsigned long timeNow;
int countThing = 0;
int GPIO_pulse = 4;                 // ESP32 = D14
float rpm, rotasi_per_detik;        // rotasi/detik
float kecepatan_kilometer_per_jam;  // kilometer/jam
float kecepatan_meter_per_detik = 0;    //meter/detik
volatile boolean flag = false;

// Curah Hujan
const int pin_interrupt = 18;  // Menggunakan pin interrupt https://www.arduino.cc/reference/en/language/functions/external-interrupts/attachinterrupt/
long int jumlah_tip = 0;
long int temp_jumlah_tip = 0;
float curah_hujan = 0.00;
float milimeter_per_tip = 0.40;

volatile boolean flag2 = false;

BH1750 lightMeter;  // BH1750 (Intensitas Cahaya)
int intensitas_cahaya =0;
float pressure =0;

// TDS & EC
#define TdsSensorPin 36
#define VREF 3.3   // analog reference voltage(Volt) of the ADC
#define SCOUNT 30  // sum of sample point

int analogBuffer[SCOUNT];  // store the analog value in the array, read from ADC
int analogBufferTemp[SCOUNT];
int analogBufferIndex = 0;
int copyIndex = 0;

float averageVoltage = 0;
float tdsValue = 0;
float ecValue = 0;


//BMP 280 Tekanan
#define BMP_SCK  (13)
#define BMP_MISO (12)
#define BMP_MOSI (11)
#define BMP_CS   (10)
Adafruit_BMP280 bmp; // I2C

// MQ-135 sensor Gas
const int mq135Pin = 39; // Pin ADC di ESP32
int adcValue =0;

//RElay  (aktuator sesuai manuscript sec. 3.6 / Algorithm 1 step 15)
const int relayPin  = 19; // UV lamp / lampu perangkap UV
const int relayPin2 = 23; // ultrasonic pest repeller / pengusir hama ultrasonik
const int relayPin3 = 27; // irrigation valve / notifikasi irigasi

// RS485
void enableTransmit() {
  digitalWrite(DE_RE_PIN, HIGH);
}

void enableReceive() {
  digitalWrite(DE_RE_PIN, LOW);
}

void flushSerialBuffer() {
  while (SerialPort.available()) SerialPort.read();
}

void ICACHE_RAM_ATTR rpm_anemometer()  // anemometer parameters (Kecepatan Angin)
{
  flag = true;
}

void ICACHE_RAM_ATTR hitung_curah_hujan()  // Curah Hujan
{
  flag2 = true;
}

// TDS dan EC
// median filtering algorithm
int getMedianNum(int bArray[], int iFilterLen) {
  int bTab[iFilterLen];
  for (byte i = 0; i < iFilterLen; i++)
    bTab[i] = bArray[i];
  int i, j, bTemp;
  for (j = 0; j < iFilterLen - 1; j++) {
    for (i = 0; i < iFilterLen - j - 1; i++) {
      if (bTab[i] > bTab[i + 1]) {
        bTemp = bTab[i];
        bTab[i] = bTab[i + 1];
        bTab[i + 1] = bTemp;
      }
    }
  }
  if ((iFilterLen & 1) > 0) {
    bTemp = bTab[(iFilterLen - 1) / 2];
  } else {
    bTemp = (bTab[iFilterLen / 2] + bTab[iFilterLen / 2 - 1]) / 2;
  }
  return bTemp;

  // TDS dan EC
  pinMode(TdsSensorPin, INPUT);
}


void setup() {
  Serial.begin(115200);
  analogReadResolution(12); // Resolusi ADC 12-bit (0 - 4095)
  
  // RS 485
  SerialPort.begin(4800, SERIAL_8N1, 16, 17); // UART2 ESP32 (RX=16, TX=17)
  pinMode(DE_RE_PIN, OUTPUT);
  pinMode(sekat, OUTPUT);
  digitalWrite(sekat, HIGH);
  enableReceive(); // Mulai dengan mode terima

  Serial.println("Membaca sensor NPK + tanah via RS485...");

  // WiFiManager
  WiFiManager wifiManager;
  wifiManager.autoConnect("Sensor_Petani_Asik");

  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Hubungkan ke WiFi
  //WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Gagal terhubung ke WiFi. Mengulangi...");
  }
  Serial.println("Terhubung ke WiFi!");
  
  dht.begin();  // DHT 22 (Suhu dan Kelembapan Udara)
  // anemometer parameters (Kecepatan Angin)
  pinMode(GPIO_pulse, INPUT_PULLUP);
  digitalWrite(GPIO_pulse, LOW);
  detachInterrupt(digitalPinToInterrupt(GPIO_pulse));                          // memulai Interrupt pada nol
  attachInterrupt(digitalPinToInterrupt(GPIO_pulse), rpm_anemometer, RISING);  //Inisialisasi pin interupt
  rpmcount = 0;
  rpm = 0;
  timeold = 0;
  timeNow = 0;
  // Curah Hujan
  pinMode(pin_interrupt, INPUT);
  attachInterrupt(digitalPinToInterrupt(pin_interrupt), hitung_curah_hujan, FALLING);  // Akan menghitung tip jika pin berlogika dari HIGH ke LOW
  // BH1750 (Intensitas Cahaya)
  Wire.begin();
  lightMeter.begin();

  // BMP tekanan
  while ( !Serial ) delay(100);   // wait for native usb
  unsigned status;
  status = bmp.begin();
  if (!status) {
    // Serial.println(F("Could not find a valid BMP280 sensor, check wiring or "
    //                   "try a different address!"));
    // Serial.print("SensorID was: 0x"); Serial.println(bmp.sensorID(),16);
    // Serial.print("        ID of 0xFF probably means a bad address, a BMP 180 or BMP 085\n");
    // Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
    // Serial.print("        ID of 0x60 represents a BME 280.\n");
    // Serial.print("        ID of 0x61 represents a BME 680.\n");
    while (1) delay(10);}
  /* Default settings from datasheet. */
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     /* Operating Mode. */
                  Adafruit_BMP280::SAMPLING_X2,     /* Temp. oversampling */
                  Adafruit_BMP280::SAMPLING_X16,    /* Pressure oversampling */
                  Adafruit_BMP280::FILTER_X16,      /* Filtering. */
                  Adafruit_BMP280::STANDBY_MS_500); /* Standby time. */ 

  //RElay
  pinMode(relayPin, OUTPUT);
  pinMode(relayPin2, OUTPUT);
  pinMode(relayPin3, OUTPUT);
  digitalWrite(relayPin,  RELAY_OFF);
  digitalWrite(relayPin2, RELAY_OFF);
  digitalWrite(relayPin3, RELAY_OFF);

  // MQTT broker aktuator
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(768);   // payload inferensi ~300 B > default 256 B
}

void loop() {
  baca_sensor();      // 1 siklus akuisisi + upload (per 15 menit)
  kontrol();          // baca status LiDAR (intruder loop)
  aktuasi();          // tarik keputusan aktuasi model fusi (REST)

  // Tunggu ~15 menit TANPA blocking penuh: tetap layani MQTT + polling aktuasi
  unsigned long tStart = millis();
  while (millis() - tStart < 900000UL) {
    if (WiFi.status() == WL_CONNECTED) {
      if (!mqtt.connected() && millis() - lastMqttReconnect > 5000) {
        lastMqttReconnect = millis();
        mqttReconnect();
      }
      mqtt.loop();
      if (millis() - lastAktuasiPoll > 15000) {   // fallback REST tiap 15 dtk
        lastAktuasiPoll = millis();
        aktuasi();
        kontrol();
      }
    }
    applyActuators();
    delay(2000);
  }
}

// ================= AKTUASI (Algorithm 1 step 15-16) =================
// Gabungkan perintah model fusi (MQTT/REST) dengan aturan lokal, lalu set relay.
void applyActuators() {
  bool uv         = cmd_uv         || local_uv_night;
  bool ultrasonic = cmd_ultrasonic || local_ultrasonic_lidar;
  bool irrigation = cmd_irrigation;
  digitalWrite(relayPin,  uv         ? RELAY_ON : RELAY_OFF);
  digitalWrite(relayPin2, ultrasonic ? RELAY_ON : RELAY_OFF);
  digitalWrite(relayPin3, irrigation ? RELAY_ON : RELAY_OFF);
}

// Terjemahkan string "action" (Algorithm 1 step 15) -> perintah relay
void setActuationFromAction(const String& action) {
  cmd_uv = cmd_ultrasonic = cmd_irrigation = false;
  if (action == "activate_ultrasonic_repeller_and_uv_lamp") {
    cmd_uv = true; cmd_ultrasonic = true;
  } else if (action == "irrigation_notification") {
    cmd_irrigation = true;
  } else if (action == "recommend_selective_insecticide_spraying") {
    Serial.println("[aktuasi] Rekomendasi: penyemprotan insektisida selektif (tanpa aktuator).");
  } // "continue_monitoring" / lainnya -> semua OFF
  Serial.print("[aktuasi] action="); Serial.print(action);
  Serial.print(" | UV="); Serial.print(cmd_uv);
  Serial.print(" ULTRA="); Serial.print(cmd_ultrasonic);
  Serial.print(" IRIG="); Serial.println(cmd_irrigation);
  applyActuators();
}

// --- MQTT: pesan taniverse/{iot_id}/inference dari HybridInferenceController ---
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<768> doc;
  if (deserializeJson(doc, payload, length)) return;
  const char* action = doc["actuation"]["action"] | "";
  if (strlen(action) > 0) setActuationFromAction(String(action));
}

void mqttReconnect() {
  String cid = "esp32-" + iot_id + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  if (mqtt.connect(cid.c_str())) {
    mqtt.subscribe(MQTT_TOPIC.c_str());
    Serial.println("[MQTT] tersambung, subscribe " + MQTT_TOPIC);
  }
}

// --- REST fallback: GET /api/inference/{iot_id} ---
void aktuasi() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(serverInference);
  int code = http.GET();
  if (code == 200) {
    StaticJsonDocument<1024> doc;
    if (!deserializeJson(doc, http.getString())) {
      const char* action = doc["data"]["actuation_action"] | "";
      if (strlen(action) > 0) setActuationFromAction(String(action));
    }
  }
  http.end();
}

void kontrol(){
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverGet);
    
    int httpResponseCode = http.GET();

    if (httpResponseCode == 200) {
      String payload = http.getString();
      Serial.println("Respon JSON:");
      Serial.println(payload);

      StaticJsonDocument<1024> doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (!error) {
        const char* status = doc["data"][0]["movement_detected"];

        // LiDAR intruder loop -> pengusir ultrasonik (dikompose di applyActuators())
        local_ultrasonic_lidar = (strcmp(status, "ON") == 0);
        applyActuators();

      } else {
        Serial.print("JSON Parsing error: ");
        Serial.println(error.c_str());
      }
    } else {
      Serial.print("HTTP GET gagal, kode: ");
      Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("WiFi belum terkoneksi...");
  }

}


void baca_sensor() {
  // printNPKValues
  digitalWrite(sekat, HIGH);
  delay(1000);
  flushSerialBuffer();
  enableTransmit();
  SerialPort.write(npkQuery, sizeof(npkQuery));
  SerialPort.flush();
  enableReceive();
  delay(500);

  if (SerialPort.available() >= 11) {
    uint8_t npkResponse[11];
    SerialPort.readBytes(npkResponse, sizeof(npkResponse));

    nitrogen   = (npkResponse[3] << 8) | npkResponse[4];
    phosphorus = (npkResponse[5] << 8) | npkResponse[6];
    potassium  = (npkResponse[7] << 8) | npkResponse[8];

    nitrogen2   = nitrogen;
    phosphorus2 = phosphorus;
    potassium2  = potassium;

    // Serial.println("=== NPK ===");
    Serial.print("Nitrogen (N): ");   Serial.print(nitrogen2);   Serial.println(" mg/kg");
    Serial.print("Phosphorus (P): "); Serial.print(phosphorus2); Serial.println(" mg/kg");
    Serial.print("Potassium (K): ");  Serial.print(potassium2);  Serial.println(" mg/kg");
  } else {
    Serial.println("Gagal membaca NPK.");
  }

  //printPHValue
  digitalWrite(sekat, LOW);
  delay(1000);
  flushSerialBuffer();
  enableTransmit();
  SerialPort.write(phQuery, sizeof(phQuery));
  SerialPort.flush();
  enableReceive();
  delay(500);

  if (SerialPort.available() >= 7) {
    uint8_t phResponse[7];
    SerialPort.readBytes(phResponse, sizeof(phResponse));

    phRaw = (phResponse[3] << 8) | phResponse[4];
    pH = phRaw / 100.0;

    // Serial.println("=== pH Tanah ===");
    Serial.print("pH: "); Serial.println(pH);
  } else {
    Serial.println("Gagal membaca pH.");
  }

  //printSoilMoisture
  digitalWrite(sekat, LOW);
  delay(1000);
  flushSerialBuffer();
  enableTransmit();
  SerialPort.write(soilMoistureQuery, sizeof(soilMoistureQuery));
  SerialPort.flush();
  enableReceive();
  delay(500);

  if (SerialPort.available() >= 7) {
    uint8_t moistureResponse[7];
    SerialPort.readBytes(moistureResponse, sizeof(moistureResponse));

    moistureRaw = (moistureResponse[3] << 8) | moistureResponse[4];
    moisture = moistureRaw / 10.0;

    // Serial.println("=== Kelembapan Tanah ===");
    Serial.print("Soil Moisture: "); Serial.print(moisture); Serial.println(" %");
  } else {
    Serial.println("Gagal membaca kelembapan.");
  }

  // printSoilTemperature
  digitalWrite(sekat, LOW);
  delay(1000);
  flushSerialBuffer();
  enableTransmit();
  SerialPort.write(soilTemperatureQuery, sizeof(soilTemperatureQuery));
  SerialPort.flush();
  enableReceive();
  delay(500);

  if (SerialPort.available() >= 7) {
    uint8_t temperatureResponse[7];
    SerialPort.readBytes(temperatureResponse, sizeof(temperatureResponse));

    tempRaw = (temperatureResponse[3] << 8) | temperatureResponse[4];
    temperature = tempRaw / 10.0;

    // Serial.println("=== Suhu Tanah ===");
    Serial.print("Soil Temperature: "); Serial.print(temperature); Serial.println(" °C");
  } else {
    Serial.println("Gagal membaca suhu tanah.");
  }

  // Sinyal
  rssi = WiFi.RSSI(); // dBm
  quality = getSignalQuality(rssi);

  Serial.print("Kekuatan Sinyal WiFi: ");
  // Serial.print(rssi);
  // Serial.print(" dBm (");
  Serial.print(quality);
  Serial.println(" %)");

  int adcValue2 = analogRead(adcPin); // Baca nilai ADC
  float vADC = (adcValue2 / maxADC) * vRef; // Tegangan hasil pembagi
  float vInput = vADC * ((R1 + R2) / R2); // Tegangan sebenarnya sebelum pembagi
  baterai = (vInput / 16.8 ) * 100;
  // Serial.print("Nilai ADC: ");
  // Serial.print(adcValue2);
  // Serial.print(" | Tegangan ADC: ");
  // Serial.print(vADC, 3);
  // Serial.print(" V | Tegangan Input: ");
  Serial.print(vInput, 2);
  Serial.println(" V");
  Serial.print("Battery: ");
  Serial.print(baterai, 2);
  Serial.println(" %");


  // DHT 22 (Suhu dan Kelembapan Udara)
  kelembapan_udara = dht.readHumidity();
  suhu_udara = dht.readTemperature();

  if (isnan(kelembapan_udara) || isnan(suhu_udara)) {
    kelembapan_udara = 0;
    suhu_udara = 0;
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  Serial.print("kelembapan_udara: ");
  Serial.println(kelembapan_udara);
  Serial.print("suhu_udara: ");
  Serial.println(suhu_udara);

  // anemometer parameters (Kecepatan Angin)
  // Cek apakah flag bernilai true, yang menandakan bahwa kondisi tertentu telah terpenuhi
  if (flag == true) {
    // Cek apakah waktu sejak deteksi terakhir sudah lebih dari atau sama dengan 5000 mikrodetik (5 milidetik)
    if (long(micros() - last_micros) >= 5000) {
      // Jika ya, maka tambahkan satu ke penghitung rpm (rpmcount)
      rpmcount++;
      // Perbarui waktu terakhir ketika magnet terdeteksi dengan waktu saat ini
      last_micros = micros();
    }
    // Setelah memproses, setel flag kembali ke false untuk menandakan bahwa kondisi telah ditangani
    flag = false;  // reset flag
  }

  if ((millis() - timeold) >= timemeasure * 1000) {
    countThing++;
    detachInterrupt(digitalPinToInterrupt(GPIO_pulse));       // Menonaktifkan interrupt saat menghitung
    rotasi_per_detik = float(rpmcount) / float(timemeasure);  // rotasi per detik
    //kecepatan_meter_per_detik = rotasi_per_detik; // rotasi/detik sebelum dikalibrasi untuk dijadikan meter per detik
    kecepatan_meter_per_detik = ((-0.0181 * (rotasi_per_detik * rotasi_per_detik)) + (1.3859 * rotasi_per_detik) + 1.4055);  // meter/detik sesudah dikalibrasi dan sudah dijadikan meter per detik
    if (kecepatan_meter_per_detik <= 1.5) {                                                                                  // Minimum pembacaan sensor kecepatan angin adalah 1.5 meter/detik
      kecepatan_meter_per_detik = 0.0;
    }
    kecepatan_kilometer_per_jam = kecepatan_meter_per_detik * 3.6;  // kilometer/jam
    // Serial.print("rotasi_per_detik=");
    // Serial.print(rotasi_per_detik);
    // Serial.print("   kecepatan_meter_per_detik="); // Minimal kecepatan angin yang dapat dibaca sensor adalah 4 meter/detik dan maksimum 30 meter/detik.
    // Serial.print(kecepatan_meter_per_detik);
    // Serial.print("   kecepatan_kilometer_per_jam=");
    // Serial.print(kecepatan_kilometer_per_jam);
    // Serial.println("   ");
    if (countThing == 1)  // kirim data per 10 detik sekali
    {
      //Serial.println("Mengirim data ke server");
      countThing = 0;
    }
    timeold = millis();
    rpmcount = 0;
    attachInterrupt(digitalPinToInterrupt(GPIO_pulse), rpm_anemometer, RISING);  // enable interrupt
  }

  // Serial.print("rotasi_per_detik=");
  // Serial.print(rotasi_per_detik);
  Serial.print("   kecepatan_meter_per_detik="); // Minimal kecepatan angin yang dapat dibaca sensor adalah 4 meter/detik dan maksimum 30 meter/detik.
  Serial.println(kecepatan_meter_per_detik);
  // Serial.print("   kecepatan_kilometer_per_jam=");
  // Serial.print(kecepatan_kilometer_per_jam);
  // Serial.println("   ");


  // Curah Hujan
  if (flag2 == true)  // don't really need the == true but makes intent clear for new users
  {
    curah_hujan += milimeter_per_tip;  // Akan bertambah nilainya saat tip penuh
    jumlah_tip++;
    delay(500);
    flag2 = false;  // reset flag2
  }
  curah_hujan = jumlah_tip * milimeter_per_tip;
  if ((jumlah_tip != temp_jumlah_tip))  // Print serial setiap 1 menit atau ketika jumlah_tip berubah
  {
    jumlah_tip = jumlah_tip;
    curah_hujan = (curah_hujan, 2);
  }
  temp_jumlah_tip = jumlah_tip;

  // Serial.print("Jumlah tip=");
  // Serial.print(jumlah_tip);
  // Serial.println(" kali ");
  // Serial.print("Curah hujan=");
  Serial.print(curah_hujan);
  Serial.println(" mm");


  // BH1750 (Intensitas Cahaya)
  intensitas_cahaya = lightMeter.readLightLevel();
  Serial.print("Light: ");
  Serial.print(intensitas_cahaya);
  Serial.println(" lx");
    // Aturan lokal: lampu perangkap UV menyala saat gelap ( < 100 lx )
    local_uv_night = (intensitas_cahaya < 100);
    applyActuators();


    //BMP 280 Tekanan
    // Serial.print(F("Temperature = "));
    // Serial.print(bmp.readTemperature());
    // Serial.println(" *C");

    pressure = bmp.readPressure();
    Serial.print(F("Pressure = "));
    Serial.print(bmp.readPressure());
    Serial.println(" Pa");

    // MQ 135 Sensor Gas
    adcValue = analogRead(mq135Pin); // Baca tegangan dari sensor MQ-135 (0–4095 pada ESP32)
    Serial.print("ADC MQ135: ");
    Serial.println(adcValue);

  // TDS dan EC
  static unsigned long analogSampleTimepoint = millis();
  if (millis() - analogSampleTimepoint > 40U) {  //every 40 milliseconds,read the analog value from the ADC
    analogSampleTimepoint = millis();
    analogBuffer[analogBufferIndex] = analogRead(TdsSensorPin);  //read the analog value and store into the buffer
    analogBufferIndex++;
    if (analogBufferIndex == SCOUNT) {
      analogBufferIndex = 0;
    }
  }

  static unsigned long printTimepoint = millis();
  if (millis() - printTimepoint > 800U) {
    printTimepoint = millis();
    for (copyIndex = 0; copyIndex < SCOUNT; copyIndex++) {
      analogBufferTemp[copyIndex] = analogBuffer[copyIndex];

      // read the analog value more stable by the median filtering algorithm, and convert to voltage value
      averageVoltage = getMedianNum(analogBufferTemp, SCOUNT) * (float)VREF / 4096.0;

      //temperature compensation formula: fFinalResult(25^C) = fFinalResult(current)/(1.0+0.02*(fTP-25.0));
      float compensationCoefficient = 1.0 + 0.02 * (temperature - 25.0);
      //temperature compensation
      float compensationVoltage = averageVoltage / compensationCoefficient;

      //convert voltage value to tds value
      tdsValue = (133.42 * compensationVoltage * compensationVoltage * compensationVoltage - 255.86 * compensationVoltage * compensationVoltage + 857.39 * compensationVoltage) * 0.5;
      ecValue = tdsValue / 0.65;  // Larutan pupuk atau air tanah: K ≈ 0,65–0,7
      //Serial.print("voltage:");
      //Serial.print(averageVoltage,2);
      //Serial.print("V   ");
      Serial.print("TDS Value:");
      Serial.print(tdsValue);
      Serial.println("ppm");
      Serial.print("EC Value:");
      Serial.print(ecValue);
      Serial.println("µS/cm");
    }
  }


  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("Accept", "application/json");


    // JSON data yang akan dikirim
    StaticJsonDocument<1024> jsonDoc;

    jsonDoc["iot_id"] = iot_id;
    jsonDoc["iot_token"] = token;
    jsonDoc["temperature"] = suhu_udara;
    jsonDoc["humidity"] = kelembapan_udara;
    jsonDoc["windspeed"] = kecepatan_meter_per_detik;
    jsonDoc["rainfall"] = curah_hujan;
    jsonDoc["light_intensity"] = intensitas_cahaya;
    jsonDoc["ph"] = pH;
    jsonDoc["soil_moisture"] = moisture;
    jsonDoc["ec"] = ecValue;
    jsonDoc["tds"] = tdsValue;
    jsonDoc["soil_temp"] = temperature;
    jsonDoc["pressure"] = pressure;
    jsonDoc["feromon"] = adcValue;
    jsonDoc["battery_level"] = baterai;
    jsonDoc["signal_strength"] = quality;
    jsonDoc["Nitrogen_Level"] = nitrogen2;
    jsonDoc["Phosphorus_Level"] = phosphorus2;
    jsonDoc["Potassium_Level"] = potassium2;

    String requestBody;
    serializeJson(jsonDoc, requestBody);

    int httpResponseCode = http.POST(requestBody);

    // Tampilkan respon
    Serial.print("Kode respon: ");
    Serial.println(httpResponseCode);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Respons:");
      Serial.println(response);
    } else {
      Serial.print("Gagal mengirim data. Error: ");
      Serial.println(http.errorToString(httpResponseCode));
    }

    http.end();
  } else {
    Serial.println("WiFi belum terkoneksi.");
  }

}

int getSignalQuality(long rssi) {
  // Konversi ke persen berdasarkan rentang -100 sampai -30 dBm
  int quality;
  if (rssi <= -100) {
    quality = 0;
  } else if (rssi >= -30) {
    quality = 100;
  } else {
    quality = 2 * (rssi + 80);
  }
  return quality;
}