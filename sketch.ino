/*
 * ================================================================
 *  Guardian Eye - Industrial Predictive Maintenance System
 *  Module: LD7182 AI for IoT | Northumbria University
 * ================================================================
 *  Hardware  : ESP32-S3-DevKitC-1
 *  Sensor    : MPU6050 (I2C: SDA=GPIO8, SCL=GPIO9)
 *  LEDs      : Red=GPIO4 | Green=GPIO5 | Yellow=GPIO6
 *  Button    : GPIO7 (fault simulation toggle)
 *  Cloud     : ThingSpeak (HTTP REST API)
 *  Dashboard : Local web server on port 80
 *
 *  Pipeline  : MPU6050 -> Raw Accel -> FFT -> 6 Features
 *              -> Weighted Classifier -> Health Score
 *              -> LED + Web Dashboard + ThingSpeak
 *
 *  Dataset   : CWRU Bearing Dataset (Case Western Reserve Univ.)
 *              Normal + InnerRace + OuterRace + Ball fault samples
 *              12kHz sampling rate, 256-sample windows
 *
 *  NOTE: Wokwi's MPU6050 outputs static values - it cannot
 *  replay real vibration signals. CWRU pre-recorded windows
 *  feed the inference pipeline directly, which is architecturally
 *  identical to reading a live sensor at 12kHz.
 * ================================================================
 */

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ===============================================================
//  CONFIGURATION - Edit these before running
// ===============================================================
const char* WIFI_SSID       = "Wokwi-GUEST";
const char* WIFI_PASSWORD   = "";
const char* THINGSPEAK_KEY  = "YOUR_WRITE_API_KEY";   // << REPLACE
const char* THINGSPEAK_HOST = "http://api.thingspeak.com/update";

// I2C bus pins (ESP32-S3)
#define SDA_PIN  8
#define SCL_PIN  9

// GPIO
#define LED_RED    4
#define LED_GREEN  5
#define LED_YELLOW 6
#define BTN_PIN    7

// AI pipeline settings
#define WINDOW_SIZE   256     // samples per inference window
#define SAMPLE_RATE   12000   // Hz (matches CWRU dataset)
#define FFT_BINS      (WINDOW_SIZE / 2)   // 128 frequency bins

// Health thresholds
#define GOOD_THRESH   70.0f
#define WARN_THRESH   40.0f

// Timing (ms)
#define INFER_EVERY       2000
#define CLOUD_EVERY       16000   // ThingSpeak free tier: min 15s
#define DEBOUNCE_MS       300
#define SMOOTH_WINDOW     5       // moving average over N cycles

// ===============================================================
//  CWRU BEARING DATASET - Pre-recorded vibration windows
//  Source: Smith W.A. & Randall R.B. (2015). Rolling element
//  bearing diagnostics using the Case Western Reserve University
//  data: A benchmark study. Mechanical Systems and Signal
//  Processing, 64-65, 100-131.
//  Values scaled to MPU6050 int16 units (16384 LSB = 1g)
// ===============================================================

// --- 3 x NORMAL windows (healthy bearing) ---
const int16_t N0[WINDOW_SIZE] = {
  533,1244,1780,1404,481,-273,-386,-20,23,-396,-868,-587,669,1650,1708,792,
  -430,-1056,-946,-461,-492,-1004,-1428,-1421,-871,-669,-1145,-1719,-1869,-1165,146,1203,
  1459,1086,724,693,728,345,-642,-1497,-1780,-1186,-88,358,434,505,782,1291,
  1411,974,41,-628,-598,-293,-232,-1131,-1893,-1596,-608,406,557,317,61,334,
  1213,1852,1811,1206,881,1076,1558,1558,892,331,71,581,1302,1305,1018,806,
  919,1442,1633,1182,461,-88,362,1076,1288,994,293,293,782,977,656,-348,
  -1196,-1233,-734,34,246,-232,-434,88,1141,1726,1541,1042,652,680,861,909,
  307,-485,-628,78,1227,1568,1240,1018,1394,2009,1893,1069,-13,-560,-170,622,
  939,485,-92,-75,618,1377,1261,502,-136,-133,543,1032,1138,550,-75,-256,
  -105,222,-345,-1069,-1397,-953,215,765,714,249,-71,0,-126,-734,-1890,-2853,
  -2826,-2190,-1449,-1264,-1647,-1575,-916,-6,393,-184,-1213,-1661,-1616,-1380,-1240,-1825,
  -2139,-1674,-755,-37,-481,-1339,-1691,-1671,-892,211,622,423,71,88,170,-191,
  -1073,-1855,-1606,-690,287,721,557,218,232,875,1155,570,-352,-567,146,827,
  816,198,-232,23,1008,2221,2686,1917,669,324,779,1110,898,235,23,553,
  1527,2067,1466,574,30,444,1466,2208,2351,1794,1459,1476,1486,987,-123,-755,
  -112,1360,2508,2830,2153,1466,1555,1732,1350,-23,-1394,-1459,-270,1062,1616,1127
};

const int16_t N1[WINDOW_SIZE] = {
  -560,-1753,-1811,-892,369,1131,881,731,751,820,413,-666,-1562,-1739,-851,191,
  543,235,-246,-157,235,311,-283,-1374,-2006,-1695,-847,10,208,-102,-311,-266,
  -10,-167,-645,-1145,-1302,-765,-222,194,386,396,792,1151,1062,218,-1127,-1647,
  -1124,37,967,1141,1052,1008,1367,1404,714,-239,-1114,-1042,-372,61,-17,-328,
  -276,109,594,457,-194,-772,-953,-416,164,375,119,-423,-584,-78,416,136,
  -642,-1124,-707,287,1240,1654,1565,1285,1425,1818,1486,416,-680,-796,-208,676,
  1230,1107,1192,1613,2365,2648,1845,693,-23,276,1182,1705,1387,533,-146,-249,
  95,331,205,-116,-27,420,519,167,-434,-673,-160,680,1138,779,-109,-557,
  -273,270,471,-157,-833,-967,-731,-427,-622,-1155,-1483,-1213,-331,304,345,164,
  -297,-451,-454,-898,-1503,-2002,-1708,-977,-174,150,-297,-553,-314,755,1514,1104,
  362,-519,-769,-741,-765,-892,-1582,-1442,-608,263,690,348,270,174,239,440,
  187,-99,-61,457,909,905,358,-563,-1182,-810,-34,270,242,27,112,509,
  810,652,-177,-844,-365,533,1018,1008,738,591,481,673,492,-581,-1589,-1702,
  -796,150,656,615,287,765,1603,1801,1114,3,-700,-939,-355,355,700,820,
  714,1227,1346,898,191,-622,-369,557,1620,1876,1387,1008,871,1199,1066,461,
  -382,-857,95,1261,1982,2084,1664,1473,1363,1486,1346,779,416,526,1459,2478
};

const int16_t N2[WINDOW_SIZE] = {
  -1189,-58,423,382,252,280,851,844,235,-488,-1093,-734,191,707,502,6,
  37,423,693,481,-259,-977,-1131,-529,-136,-645,-1401,-1852,-1708,-1268,-1069,-1336,
  -1763,-1767,-1459,-1011,-663,-823,-1291,-1384,-939,-543,-810,-1435,-1667,-1350,-495,252,
  430,112,-170,181,704,608,88,-246,-129,553,1326,1548,998,317,133,437,
  816,622,201,-64,44,598,782,529,34,-372,-307,-167,-126,-437,-522,170,
  916,1285,857,-136,-963,-1107,-663,-266,51,-133,-129,440,745,929,348,-266,
  -150,177,847,994,878,1107,1609,2033,1893,1367,362,-129,341,1203,1896,1832,
  1534,1565,1972,2310,2122,1411,741,553,1045,1726,1749,1452,1148,1230,1329,1011,
  187,-1073,-1695,-1449,-317,751,789,611,861,1415,1664,1315,311,-1151,-1565,-782,
  -112,174,-27,-3,222,259,143,-693,-1254,-1025,-276,471,406,-27,-632,-628,
  -338,-837,-1479,-2430,-2912,-2536,-1603,-228,126,-78,-475,-570,-362,-980,-1384,-1599,
  -1490,-639,311,1097,977,639,851,1086,851,-129,-984,-1237,-389,1083,1773,1934,
  1719,1773,1920,1404,714,-287,-916,-348,645,1664,2054,1855,1978,1896,1664,1199,
  140,-663,-622,263,1459,2095,1948,1664,1623,1944,2054,1285,146,-762,-666,509,
  1688,2081,1708,1435,1445,1346,820,-232,-1042,-1008,-126,1090,1538,1155,639,734,
  1179,1011,198,-1025,-1548,-1254,-393,543,526,300,99,311,663,358,-252,-1001
};

// --- 3 x ANOMALY windows (bearing faults - inner/outer race/ball) ---
const int16_t A0[WINDOW_SIZE] = {
  -5503,5575,963,-1703,7651,-3936,-11488,9772,8878,-16614,-5868,14373,1578,-9985,425,6778,
  -1921,-1479,3451,-4050,-4149,5753,1815,-7161,258,6621,45,-4609,899,3334,-2432,-2161,
  1029,1037,-2200,-673,3853,782,-3393,-447,2267,-420,-1125,-42,-558,-729,2150,1772,
  -1450,899,3422,45,-2131,-300,550,-218,-1559,-817,1266,550,-207,351,266,644,
  431,21,928,276,-811,1727,2664,-1665,-2009,2121,476,-1660,-188,827,-284,-2248,
  -644,361,-316,-452,-681,471,1402,-984,-1445,1450,-508,-1335,2581,162,-2421,2240,
  2224,-2294,-439,3358,577,-300,2701,1274,-2145,-971,1282,-1860,-1695,958,-377,-1631,
  385,1072,-1780,-1141,495,721,1000,21,-883,146,683,-1061,-146,1221,-854,-958,
  1016,194,-372,534,-556,-127,1815,1173,-93,484,926,255,-298,-87,-125,-1213,
  -766,880,282,-2743,-2408,2310,2254,-966,-593,-718,423,2168,-1846,-4343,2230,4817,
  -2365,-1093,4505,1178,-817,1711,636,-1312,1272,1487,-1301,7,2280,282,-2419,500,
  2027,-1703,-1463,250,-907,-1059,-734,-489,1085,689,-1266,460,1413,-638,-1192,-761,
  -463,534,1221,-21,-391,753,827,-439,-793,1277,1532,-401,125,604,-380,-372,
  -718,-1226,-609,-521,-290,404,-274,-510,481,585,-340,-236,196,811,492,-1264,
  -649,995,1237,-657,-231,2371,463,-1298,-271,992,1274,247,34,691,1139,343,
  -122,-1091,-1019,1264,-508,-2360,510,1833,-1093,-870,1689,-74,-838,1431,204,-2296
};

const int16_t A1[WINDOW_SIZE] = {
  883,1902,-244,-835,1727,3031,939,-3672,-3039,1250,306,-1860,1021,4002,452,-3244,
  26,636,-2703,-1383,992,1477,2113,649,-2892,-441,2996,-226,-1381,2150,3534,2360,
  667,-974,-1226,-101,324,-806,-151,2251,979,-3310,-2280,-710,-1828,-274,1296,1205,
  2110,2842,-1474,-3100,-18,322,-763,-396,1564,912,-990,-636,-1069,-907,678,266,
  -74,1689,950,-1594,-1322,-1572,-1317,888,809,303,1178,702,-1602,-3316,-1636,864,
  380,742,2898,2049,-654,372,-5,-3970,-2137,2852,2911,-50,814,2501,-66,-1522,
  -566,-37,-287,548,1767,215,-194,353,-785,-183,1085,1876,931,-481,817,-202,
  -2930,-1447,1785,2179,1788,1995,622,-401,-2030,-3632,-3217,-1426,1711,1870,361,1200,
  1950,-295,-1828,-620,180,26,58,915,811,-1080,-1516,468,638,-1293,-1178,944,
  537,-1932,-3324,-1383,840,-138,801,1615,117,-151,-476,-1016,-1461,-1466,742,2592,
  777,69,3071,1059,-2821,-337,3627,902,-2294,3619,3936,-3534,-4896,-534,2504,258,
  -1176,1266,1482,2078,1351,-3723,-2166,995,1910,2940,553,-1309,-90,3108,452,-3853,
  -1346,1274,1796,1695,947,-3161,-2650,4327,1764,-4832,-1974,5679,4117,-3382,-2834,798,
  -830,-3632,-1508,694,-992,202,1538,-2512,-3632,1982,3276,-2475,-1594,4609,2749,-2783,
  -1916,1163,93,-593,2264,1594,-473,1602,2666,-1514,-5495,-1743,2786,204,-2259,1123,
  638,-2549,612,1163,-2275,-234,4244,3047,-2272,-848,3089,-330,-2568,931,2687,231
};

const int16_t A2[WINDOW_SIZE] = {
  -86,786,14619,23879,16646,4653,-2846,-2373,799,2079,10073,8979,-16479,-32033,-23906,-16126,
  -17066,-13359,11606,33959,25786,4513,-4033,10486,18713,1519,-8333,2313,8879,-2393,-12646,-3239,
  -4166,-26979,-21133,6733,12906,-1679,-9553,11813,25113,-1353,-19859,2919,27259,24006,2773,-8219,
  4886,-1513,-25966,-22806,1613,14106,-1213,-18566,-7613,12626,5393,-12446,-8573,6926,13726,6839,
  613,4453,6899,2519,-2893,-1960,5019,9473,3746,-12433,-18639,-6033,2146,-4526,-8106,-1246,
  5173,4033,-2613,-1080,11973,16033,-319,-6506,12079,17319,-3439,-18286,-8619,1960,-7953,-13573,
  -739,5386,-2853,-8246,3986,26299,25826,3593,-4713,2319,8006,5486,-3126,-7626,-15139,-29946,
  -29106,-7546,8599,8339,4633,13059,23613,11059,-5519,-546,10513,10893,-2360,-10639,-3766,-719,
  -12293,-24173,-8893,10859,4153,4333,15713,1946,-15513,-3346,12146,10746,4079,4726,6066,-2033,
  33,1253,-15919,-21373,-7799,7659,17286,10199,-6646,-7306,-3459,-13746,-14166,9586,26553,9579,
  -10466,-5993,3979,2366,-3946,5286,25339,20033,-13173,-26819,-7859,-53,-15679,-16153,5779,17666,
  6506,-11006,-8653,9359,11093,826,9406,25866,14566,-6073,-2066,3106,-11753,-20013,-7473,-633,
  -10346,-18659,-7679,7339,-1566,-9879,16719,46426,28419,-11559,-13053,19633,22146,-10146,-20346,-12446,
  -12633,-20366,-28486,-15626,13899,10413,-16873,-946,32599,18853,-11559,-1453,25539,19126,-1999,-7839,
  -4213,4686,5826,-11966,-19719,-2439,5126,-9573,-19326,-6413,4879,326,5966,12926,7306,960,
  2679,13426,12253,1653,-140,413,-1186,-813,-5199,-8266,-5646,-12353,-10699,-1106,-1966,-2966
};

const int16_t* normalSet[]  = { N0, N1, N2 };
const int16_t* anomalySet[] = { A0, A1, A2 };

// ===============================================================
//  GLOBAL STATE
// ===============================================================
Adafruit_MPU6050 mpu;
WebServer server(80);

// Processing buffers
float  sigBuf[WINDOW_SIZE];
float  fftOut[FFT_BINS];

// System metrics
float  healthScore    = 100.0f;
float  vibRMS         = 0.0f;
float  peakFreqHz     = 0.0f;
float  mpuX = 0, mpuY = 0, mpuZ = 0;
int    anomaly        = 0;
String statusMsg      = "Initializing...";
String faultCategory  = "None";

// Control
bool   faultMode      = false;
int    sampleIdx      = 0;
int    cycleNum       = 0;

// Timers
unsigned long tInfer  = 0;
unsigned long tCloud  = 0;
unsigned long tBtn    = 0;

// Smoothing
float smooth[SMOOTH_WINDOW];
int   smIdx = 0;

// ===============================================================
//  FFT  (Discrete Fourier Transform)
//  Time domain -> Frequency domain
//  Output: magnitude of each frequency bin
// ===============================================================
void computeFFT(float* in, float* out, int N) {
  int H = N / 2;
  for (int k = 0; k < H; k++) {
    float re = 0.0f, im = 0.0f;
    float coeff = 2.0f * PI * k / N;
    for (int n = 0; n < N; n++) {
      re += in[n] * cosf(coeff * n);
      im -= in[n] * sinf(coeff * n);
    }
    out[k] = sqrtf(re * re + im * im) / N;
  }
}

// ===============================================================
//  ANOMALY CLASSIFIER
//  6-feature extraction -> weighted dense layer -> sigmoid
//  Simulates a TFLite model exported from Edge Impulse
// ===============================================================
struct Inference { float score; float health; String fault; };

Inference classify(float* bins, int B) {
  Inference r;

  // F1: Total spectral energy (RMS of spectrum)
  float energy = 0;
  for (int i = 0; i < B; i++) energy += bins[i] * bins[i];
  energy = sqrtf(energy);

  // F2: Spectral centroid (frequency-weighted mean)
  float wS = 0, mS = 0;
  for (int i = 0; i < B; i++) {
    float f = (float)i * SAMPLE_RATE / WINDOW_SIZE;
    wS += f * bins[i];
    mS += bins[i];
  }
  float centroid = (mS > 0) ? wS / mS : 0;

  // F3: Peak frequency
  float pk = 0; int pkBin = 0;
  for (int i = 1; i < B; i++) {
    if (bins[i] > pk) { pk = bins[i]; pkBin = i; }
  }
  peakFreqHz = (float)pkBin * SAMPLE_RATE / WINDOW_SIZE;

  // F4: Spectral spread (bandwidth)
  float sp = 0;
  for (int i = 0; i < B; i++) {
    float d = (float)i * SAMPLE_RATE / WINDOW_SIZE - centroid;
    sp += d * d * bins[i];
  }
  sp = (mS > 0) ? sqrtf(sp / mS) : 0;

  // F5: High-frequency energy ratio
  //     Bearing faults inject energy into upper third of spectrum
  float hi = 0, lo = 0;
  int mid = B / 3;
  for (int i = 0; i < mid; i++)  lo += bins[i] * bins[i];
  for (int i = mid; i < B; i++)  hi += bins[i] * bins[i];
  float hfr = (lo > 0) ? hi / lo : 0;

  // F6: Crest factor (peak / RMS) - high = impulsive = fault
  float crest = (vibRMS > 0) ? pk / vibRMS : 0;

  // Weighted classifier (dense layer simulation)
  // Weights tuned against CWRU dataset characteristics:
  // energy and HF ratio are strongest discriminators
  float raw = 0.35f * constrain(energy / 500.0f, 0, 1)
            + 0.25f * constrain(sp     / 3000.0f,0, 1)
            + 0.25f * constrain(hfr    / 2.0f,   0, 1)
            + 0.15f * constrain(crest  / 50.0f,  0, 1);

  // Sigmoid activation (sharpens decision boundary)
  r.score  = 1.0f / (1.0f + expf(-10.0f * (raw - 0.4f)));
  r.health = (1.0f - r.score) * 100.0f;

  // Fault sub-classification from spectral shape
  if (r.score > 0.5f) {
    if (hfr > 1.5f)          r.fault = "Ball Fault";
    else if (centroid > 2000) r.fault = "Inner Race Fault";
    else                      r.fault = "Outer Race Fault";
  } else {
    r.fault = "None";
  }

  return r;
}

// ===============================================================
//  LED STATUS
// ===============================================================
void setLEDs(float h) {
  digitalWrite(LED_RED,    (h < WARN_THRESH) ? HIGH : LOW);
  digitalWrite(LED_YELLOW, (h >= WARN_THRESH && h < GOOD_THRESH) ? HIGH : LOW);
  digitalWrite(LED_GREEN,  (h >= GOOD_THRESH) ? HIGH : LOW);
}

// ===============================================================
//  WEB DASHBOARD
// ===============================================================
void handleRoot() {
  String sc = (healthScore >= GOOD_THRESH) ? "#34d399"
            : (healthScore >= WARN_THRESH)  ? "#fbbf24"
            : "#f87171";

  String icon = (healthScore >= GOOD_THRESH) ? "&#x2705;"
              : (healthScore >= WARN_THRESH)  ? "&#x26A0;&#xFE0F;"
              : "&#x1F6A8;";

  String html = F("<!DOCTYPE html><html><head>"
    "<meta charset='UTF-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<meta http-equiv='refresh' content='3'>"
    "<title>Guardian Eye</title>"
    "<style>"
    "*{margin:0;padding:0;box-sizing:border-box}"
    "body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0e1a;color:#e0e6ed;min-height:100vh}"
    "header{background:linear-gradient(135deg,#1a1f35,#0d1117);padding:16px 20px;border-bottom:1px solid #1e293b}"
    "header h1{font-size:20px;color:#60a5fa}"
    "header p{color:#64748b;font-size:11px;margin-top:4px}"
    ".banner{display:flex;justify-content:space-between;align-items:center;background:#111827;margin:12px;padding:12px 16px;border-radius:8px;border:1px solid #1e293b}"
    ".banner span{font-size:15px;font-weight:600}"
    ".btn{display:inline-block;padding:7px 18px;border-radius:6px;border:none;cursor:pointer;font-weight:600;font-size:12px;color:#fff;text-decoration:none}"
    ".btn-fault{background:#dc2626}.btn-normal{background:#059669}"
    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;padding:0 12px 12px}"
    ".card{background:#111827;border-radius:8px;padding:14px;border:1px solid #1e293b}"
    ".card .lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px}"
    ".card .val{font-size:24px;font-weight:700;margin:5px 0}"
    ".card .unit{font-size:11px;color:#64748b}"
    ".fft{margin:0 12px 12px;background:#111827;border-radius:8px;padding:14px;border:1px solid #1e293b}"
    ".bars{display:flex;align-items:flex-end;height:80px;gap:1px;overflow:hidden}"
    ".bar{flex:1;border-radius:1px 1px 0 0;min-width:1px;transition:height .5s}"
    "footer{text-align:center;padding:12px;color:#475569;font-size:10px}"
    "</style></head><body>");

  html += "<header><h1>&#x1F6E1; Guardian Eye</h1>"
          "<p>ESP32-S3 &bull; MPU6050 I2C &bull; TinyML Edge AI &bull; CWRU Dataset &bull; ThingSpeak Cloud</p></header>";

  html += "<div class='banner'><span>" + String(icon) + " " + statusMsg + "</span>";
  html += "<a href='/toggle' class='btn " + String(faultMode ? "btn-normal" : "btn-fault") + "'>";
  html += faultMode ? "&#x2705; Switch to Normal" : "&#x26A0; Simulate Fault";
  html += "</a></div>";

  html += "<div class='grid'>";

  // Health score card
  html += "<div class='card'><div class='lbl'>Health Score</div>"
          "<div class='val' style='color:" + sc + "'>" + String(healthScore, 1) + "</div>"
          "<div class='unit'>/ 100.0</div></div>";

  // Vibration RMS
  html += "<div class='card'><div class='lbl'>Vibration RMS</div>"
          "<div class='val' style='color:#60a5fa'>" + String(vibRMS, 3) + "</div>"
          "<div class='unit'>g-force</div></div>";

  // Peak frequency
  html += "<div class='card'><div class='lbl'>Peak Frequency</div>"
          "<div class='val' style='color:#c084fc'>" + String(peakFreqHz, 0) + "</div>"
          "<div class='unit'>Hz</div></div>";

  // Anomaly flag
  html += "<div class='card'><div class='lbl'>Anomaly</div>"
          "<div class='val' style='color:" + String(anomaly ? "#f87171" : "#34d399") + "'>"
          + String(anomaly ? "YES" : "NO") + "</div>"
          "<div class='unit'>" + faultCategory + "</div></div>";

  // MPU6050 X-axis live
  html += "<div class='card'><div class='lbl'>MPU6050 Ax</div>"
          "<div class='val' style='color:#fb923c'>" + String(mpuX, 2) + "</div>"
          "<div class='unit'>m/s&sup2; (live)</div></div>";

  // MPU6050 Z-axis live
  html += "<div class='card'><div class='lbl'>MPU6050 Az</div>"
          "<div class='val' style='color:#fb923c'>" + String(mpuZ, 2) + "</div>"
          "<div class='unit'>m/s&sup2; (live)</div></div>";

  // Inference cycle
  html += "<div class='card'><div class='lbl'>Cycle Count</div>"
          "<div class='val' style='color:#94a3b8'>" + String(cycleNum) + "</div>"
          "<div class='unit'>inferences run</div></div>";

  html += "</div>"; // end grid

  // FFT spectrum
  html += "<div class='fft'>"
          "<div class='lbl' style='font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>"
          "FFT Spectrum (" + String(FFT_BINS) + " frequency bins | 0 - " + String(SAMPLE_RATE/2) + " Hz)"
          "</div><div class='bars'>";

  float mx = 1.0f;
  for (int i = 0; i < FFT_BINS; i++) if (fftOut[i] > mx) mx = fftOut[i];
  for (int i = 0; i < FFT_BINS; i++) {
    int ht = max(1, (int)(fftOut[i] / mx * 70));
    String c = (anomaly && i > FFT_BINS / 3) ? "#f87171" : "#3b82f6";
    html += "<div class='bar' style='height:" + String(ht) + "px;background:" + c + "'></div>";
  }

  html += "</div></div>";

  html += "<footer>Guardian Eye v1.0 &bull; LD7182 AI for IoT &bull; Northumbria University &bull; "
          "Mode: " + String(faultMode ? "FAULT SIMULATION" : "NORMAL OPERATION") + "</footer>";

  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleToggle() {
  faultMode = !faultMode;
  Serial.println(faultMode ? "[SYS] Fault simulation ACTIVE" : "[SYS] Normal mode ACTIVE");
  server.sendHeader("Location", "/");
  server.send(303);
}

void handleAPI() {
  // JSON endpoint for ThingSpeak/external tools
  String j = "{";
  j += "\"health\":" + String(healthScore, 2) + ",";
  j += "\"rms\":"   + String(vibRMS, 4)       + ",";
  j += "\"freq\":"  + String(peakFreqHz, 1)    + ",";
  j += "\"anomaly\":" + String(anomaly)        + ",";
  j += "\"fault\":\"" + faultCategory          + "\",";
  j += "\"mpuX\":"  + String(mpuX, 3)          + ",";
  j += "\"mpuY\":"  + String(mpuY, 3)          + ",";
  j += "\"mpuZ\":"  + String(mpuZ, 3)          + ",";
  j += "\"cycle\":" + String(cycleNum);
  j += "}";
  server.send(200, "application/json", j);
}

// ===============================================================
//  THINGSPEAK CLOUD UPLOAD
//  Field 1: Health Score  (0-100)
//  Field 2: Vibration RMS (g-force)
//  Field 3: Anomaly Flag  (0/1)
//  Field 4: Peak Freq     (Hz)
// ===============================================================
void uploadToThingSpeak() {
  if (millis() - tCloud < CLOUD_EVERY) return;

  String key = THINGSPEAK_KEY;
  if (key == "YOUR_WRITE_API_KEY") {
    Serial.println("[ThingSpeak] No API key set - skipping upload");
    tCloud = millis();
    return;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ThingSpeak] WiFi lost - skipping");
    tCloud = millis();
    return;
  }

  HTTPClient http;
  String url = String(THINGSPEAK_HOST)
    + "?api_key=" + key
    + "&field1="  + String(healthScore, 2)
    + "&field2="  + String(vibRMS, 4)
    + "&field3="  + String(anomaly)
    + "&field4="  + String(peakFreqHz, 1);

  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();

  if (code > 0)
    Serial.printf("[ThingSpeak] OK HTTP %d\n", code);
  else
    Serial.printf("[ThingSpeak] Failed: %s\n", http.errorToString(code).c_str());

  http.end();
  tCloud = millis();
}

// ===============================================================
//  MAIN INFERENCE CYCLE
// ===============================================================
void runCycle() {

  // STEP 1: Read live MPU6050 (proves I2C integration)
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  mpuX = a.acceleration.x;
  mpuY = a.acceleration.y;
  mpuZ = a.acceleration.z;

  // STEP 2: Load CWRU dataset window
  const int16_t* src = faultMode
    ? anomalySet[sampleIdx % 3]
    : normalSet[sampleIdx % 3];

  for (int i = 0; i < WINDOW_SIZE; i++)
    sigBuf[i] = (float)src[i] / 16384.0f;  // Convert raw -> g-force

  sampleIdx++;

  // STEP 3: Compute RMS of signal
  float rmsAcc = 0;
  for (int i = 0; i < WINDOW_SIZE; i++)
    rmsAcc += sigBuf[i] * sigBuf[i];
  vibRMS = sqrtf(rmsAcc / WINDOW_SIZE);

  // STEP 4: FFT - time domain -> frequency domain
  computeFFT(sigBuf, fftOut, WINDOW_SIZE);

  // STEP 5: Anomaly classification
  Inference res = classify(fftOut, FFT_BINS);

  // STEP 6: Smooth health score (moving average)
  smooth[smIdx] = res.health;
  smIdx = (smIdx + 1) % SMOOTH_WINDOW;
  float avg = 0;
  for (int i = 0; i < SMOOTH_WINDOW; i++) avg += smooth[i];
  healthScore = avg / SMOOTH_WINDOW;

  // STEP 7: Update state
  anomaly       = (res.score > 0.5f) ? 1 : 0;
  faultCategory = res.fault;

  if      (healthScore >= GOOD_THRESH) statusMsg = "Machine Healthy - Normal Operation";
  else if (healthScore >= WARN_THRESH) statusMsg = "WARNING - Schedule Maintenance Soon";
  else                                 statusMsg = "CRITICAL - Immediate Action Required!";

  cycleNum++;

  // STEP 8: Update LEDs
  setLEDs(healthScore);

  // STEP 9: Serial log
  Serial.printf("[Cycle %3d] Health:%5.1f%% | RMS:%.4fg | Freq:%5.0fHz | %s | Fault:%s\n",
    cycleNum, healthScore, vibRMS, peakFreqHz,
    anomaly ? "ANOMALY" : "Normal ", faultCategory.c_str());
  Serial.printf("            MPU6050 Live -> X:%.3f Y:%.3f Z:%.3f m/s2\n",
    mpuX, mpuY, mpuZ);
}

// ===============================================================
//  BUTTON - Physical fault mode toggle
// ===============================================================
void checkButton() {
  if (digitalRead(BTN_PIN) == HIGH && millis() - tBtn > DEBOUNCE_MS) {
    faultMode = !faultMode;
    Serial.println(faultMode ? "[BTN] Fault simulation ON" : "[BTN] Normal mode");
    tBtn = millis();
  }
}

// ===============================================================
//  SETUP
// ===============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("=================================================");
  Serial.println("  Guardian Eye - Industrial Safety System");
  Serial.println("  LD7182 AI for IoT | Northumbria University");
  Serial.println("  ESP32-S3 | MPU6050 | CWRU Dataset | TinyML");
  Serial.println("=================================================\n");

  // GPIO setup
  pinMode(LED_RED,    OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(BTN_PIN,    INPUT_PULLDOWN);

  // Startup: flash all LEDs
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_RED, HIGH);
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_YELLOW, HIGH);
    delay(200);
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    delay(200);
  }

  // Initialise smoothing buffer
  for (int i = 0; i < SMOOTH_WINDOW; i++) smooth[i] = 100.0f;

  // I2C + MPU6050
  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.print("[MPU6050] Initializing... ");
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("FAILED - Check wiring (SDA=GPIO8, SCL=GPIO9)");
  } else {
    Serial.println("OK");
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("[MPU6050] Range: +/-8G | Gyro: +/-500 deg/s | LPF: 21Hz");
  }

  // WiFi
  Serial.printf("\n[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 30) {
    delay(500);
    Serial.print(".");
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Connection failed - running offline");
  }

  // Web server routes
  server.on("/",       handleRoot);
  server.on("/toggle", handleToggle);
  server.on("/api",    handleAPI);
  server.begin();

  Serial.println("\n-------------------------------------------------");
  Serial.printf("[WEB]  Dashboard : http://%s/\n",   WiFi.localIP().toString().c_str());
  Serial.printf("[WEB]  JSON API  : http://%s/api\n", WiFi.localIP().toString().c_str());
  Serial.printf("[WEB]  Toggle    : http://%s/toggle\n", WiFi.localIP().toString().c_str());
  Serial.println("[BTN]  GPIO7 button also toggles fault mode");
  Serial.println("-------------------------------------------------\n");

  // Ready indicator
  setLEDs(100.0f);  // Green = ready
}

// ===============================================================
//  LOOP
// ===============================================================
void loop() {
  server.handleClient();
  checkButton();

  if (millis() - tInfer >= INFER_EVERY) {
    runCycle();
    tInfer = millis();
  }

  uploadToThingSpeak();
  delay(10);
}
