// ====== Pins ======  
const int PIN_INFLATE = 3;    // Solenoid valve (只能全开/全关)
const int PIN_VALVE   = 4;    // Inflation pump (吸气泵)
const int PIN_DEFLATE = 5;    // Deflation pump (呼气泵)

// ====== UAO parameters ======
// 吸气总时间由两段 800ms 吸 + 两段 400ms 停组成，呼气 600ms
const int INHALE_ON_MS    = 1000;   // 每一小段“实际在吸”的时间
const int INHALE_HOLD_MS  = 600;   // 每一小段“停住不动”的时间
const int EXHALE_TIME_MS  = 1300;   // 呼气 600 ms

// 这里你现在是 255 / 255，如果之后想调体积，可以再改
const int INHALE_PUMP_SPEED  = 255;   // 吸气泵速度
const int EXHALE_PUMP_SPEED  = 255;   // 呼气泵速度

// 电磁阀：全开或全关
const int VALVE_FULL_OPEN = 255;
const int VALVE_CLOSED    = 0;

// 统一封装一个呼吸周期
void doUAOBreath() {
  // ===== 吸气阶段：800 吸 + 400 停 + 800 吸 + 400 停 =====
  Serial.println("Inhale segment 1 (800 ms ON) - UAO");
  analogWrite(PIN_INFLATE, VALVE_FULL_OPEN);    // 打开气道
  analogWrite(PIN_VALVE,   INHALE_PUMP_SPEED);  // 吸气泵开启
  analogWrite(PIN_DEFLATE, 0);
  delay(INHALE_ON_MS);                          // 800 ms 吸气

  Serial.println("Hold 1 (400 ms PAUSE)");
  analogWrite(PIN_INFLATE, VALVE_CLOSED);       // 关闭气道，保持体积
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, 0);
  delay(INHALE_HOLD_MS);                        // 400 ms 停住

  Serial.println("Inhale segment 2 (800 ms ON) - UAO");
  analogWrite(PIN_INFLATE, VALVE_FULL_OPEN);
  analogWrite(PIN_VALVE,   INHALE_PUMP_SPEED);
  analogWrite(PIN_DEFLATE, 0);
  delay(INHALE_ON_MS);                          // 再吸 800 ms

  Serial.println("Hold 2 (400 ms PAUSE)");
  analogWrite(PIN_INFLATE, VALVE_CLOSED);
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, 0);
  delay(INHALE_HOLD_MS);                        // 再停 400 ms

  // 总“吸气阶段”时间 = 800 + 400 + 800 + 400 = 2400 ms

  // ===== 呼气阶段（短、流量大：呼气快速）=====
  Serial.println("Exhale (short, high flow) - UAO");
  analogWrite(PIN_INFLATE, VALVE_CLOSED);       // 按你原逻辑：呼气时关电磁阀
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, EXHALE_PUMP_SPEED);
  delay(EXHALE_TIME_MS);                        // 600 ms 呼气

  // ===== 停止阶段（短暂停顿）=====
  Serial.println("Pause");
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, 0);
  delay(100);
}

void setup() {
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE,   OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);

  Serial.begin(9600);
  Serial.println("UAO breathing pattern started (segmented inhale 800/400)");

  for (int cycle = 0; cycle < 1000; cycle++) {
    Serial.print("Cycle ");
    Serial.println(cycle + 1);
    doUAOBreath();
  }

  Serial.println("All cycles completed");
}

void loop() {}
