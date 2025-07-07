#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Ganti dengan SSID dan Password WiFi kamu
const char* ssid = "Petani Asik";
const char* password = "petaniasik123";

// Ganti dengan alamat IP atau domain server Laravel kamu
const char* serverName = "http://petaniasik.local/api/kirim_sensor"; // contoh lokal

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Menghubungkan ke WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nTerhubung ke WiFi");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("Accept", "application/json");

    // Buat JSON untuk dikirim
    StaticJsonDocument<1024> jsonDoc;

    // Token dan ID harus cocok dengan data users di Laravel
    jsonDoc["iot_id"] = "jTZids5M";
    jsonDoc["iot_token"] = "lOsx0KMJSAfJpxEe";

    // Contoh data sensor dummy
    jsonDoc["temperature"] = 29.3;
    jsonDoc["humidity"] = 72;
    jsonDoc["windspeed"] = 1.4;
    jsonDoc["rainfall"] = 0.0;
    jsonDoc["light_intensity"] = 750;
    jsonDoc["ph"] = 6.8;
    jsonDoc["soil_moisture"] = 30.2;
    jsonDoc["ec"] = 1.1;
    jsonDoc["tds"] = 480;
    jsonDoc["soil_temp"] = 26.7;
    jsonDoc["pressure"] = 1008.3;
    jsonDoc["feromon"] = 0;
    jsonDoc["battery_level"] = 85;
    jsonDoc["signal_strength"] = 65;
    jsonDoc["Nitrogen_Level"] = 18.5;
    jsonDoc["Phosphorus_Level"] = 5.3;
    jsonDoc["Potassium_Level"] = 2.9;

    // Konversi JSON ke string
    String requestBody;
    serializeJson(jsonDoc, requestBody);

    // Kirim POST request
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
    Serial.println("WiFi tidak terhubung");
  }

  delay(60000); // Kirim data setiap 60 detik
}
