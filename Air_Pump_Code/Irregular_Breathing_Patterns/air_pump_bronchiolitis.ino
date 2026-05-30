// ====== Pins ======吸气短呼气长
const int PIN_INFLATE = 3;    // Solenoid valve (只能全开/全关)
const int PIN_VALVE   = 4;    // Inflation pump (吸气泵)
const int PIN_DEFLATE = 5;    // Deflation pump (呼气泵)

// ====== Bronchiolitis parameters ======
// 吸气时间短，呼气时间长（2 倍）
const int INHALE_TIME_MS = 900;   // 吸气 300 ms
const int EXHALE_TIME_MS = 2000;   // 呼气 600 ms

// 为保证吸入体积 ≈ 呼出体积：吸气流量 ≈ 呼气流量的 2 倍
// 注意上限 255，不能真的写成 2x>255
const int INHALE_PUMP_SPEED = 240;  // 吸气泵速度（较快）
const int EXHALE_PUMP_SPEED = 120;  // 呼气泵速度（较慢）

// 电磁阀：全开或全关
const int VALVE_FULL_OPEN = 255;
const int VALVE_CLOSED    = 0;

// 统一封装一个呼吸周期
void doBronchiolitisBreath() {
  // ===== 吸气阶段（短、流量大）=====
  Serial.println("Inhale (short, high flow)");
  analogWrite(PIN_INFLATE, VALVE_FULL_OPEN);   // 电磁阀全开
  analogWrite(PIN_VALVE,   INHALE_PUMP_SPEED); // 吸气泵高速
  analogWrite(PIN_DEFLATE, 0);                 // 呼气泵关闭
  delay(INHALE_TIME_MS);

  // ===== 呼气阶段（长、流量小）=====
  Serial.println("Exhale (long, low flow)");
  analogWrite(PIN_INFLATE, VALVE_CLOSED);      // 电磁阀关闭
  analogWrite(PIN_VALVE,   0);                 // 吸气泵关闭
  analogWrite(PIN_DEFLATE, EXHALE_PUMP_SPEED); // 呼气泵低速
  delay(EXHALE_TIME_MS);

  // ===== 停止阶段（短暂停顿，让系统回到平衡）=====
  Serial.println("Pause");
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, 0);
  // 可以加一点小停顿，比如 100 ms，让节律更“喘息感”
  delay(100);
}

void setup() {
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE,   OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);

  Serial.begin(9600);
  Serial.println("Bronchiolitis breathing pattern started");
  Serial.print("Inhale pump speed: ");
  Serial.print((INHALE_PUMP_SPEED / 255.0) * 100);
  Serial.println("%");
  Serial.print("Exhale pump speed: ");
  Serial.print((EXHALE_PUMP_SPEED / 255.0) * 100);
  Serial.println("%");

  for (int cycle = 0; cycle < 1000; cycle++) {
    Serial.print("Cycle ");
    Serial.println(cycle + 1);
    doBronchiolitisBreath();
  }

  Serial.println("All cycles completed");
}

void loop() {
  // 不循环，所有动作都在 setup 中完成
}
