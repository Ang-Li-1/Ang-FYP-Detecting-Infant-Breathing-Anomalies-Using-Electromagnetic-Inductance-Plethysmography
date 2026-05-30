// ===== Pins (保持与你现有硬件一致) =====
const int PIN_INFLATE = 3;   // 电磁阀（打开=允许进气）
const int PIN_VALVE   = 4;   // 充气泵
const int PIN_DEFLATE = 5;   // 泄气泵

// ===== 基础转速（0–255）=====
// 说明：电磁阀通常只有开/关（255/0）；充/放气泵可用占空比调强弱
const int SPEED_VALVE_NORMAL   = 150; // 正常小呼吸时的充气泵力度
const int SPEED_DEFLATE_NORMAL = 200; // 正常小呼吸时的放气泵力度
const int SPEED_VALVE_RECO     = 200; // 补偿性大呼吸时的充气泵力度（更强）
const int SPEED_DEFLATE_RECO   = 250; // 补偿性大呼吸时的放气泵力度（更强）
const int SPEED_SOLENOID       = 255; // 电磁阀全开

// ===== AOP（周期性暂停）参数 =====
// 正常呼吸节律（早产儿常见：较快、较浅，这里取 ~40–50 BPM 的节拍）
unsigned int INHALE_MS_NORMAL   = 650;   // 正常吸气时长
unsigned int EXHALE_MS_NORMAL   = 750;   // 正常呼气时长
unsigned int N_NORMAL_BEFORE_AP = 8;     // 每次暂停前先做多少个正常呼吸

// 暂停（平线段）
bool         RANDOM_APNEA       = true;  // 是否在范围内随机暂停时长
unsigned int APNEA_MS_MIN       = 8000;  // 暂停最少 8 s
unsigned int APNEA_MS_MAX       = 12000; // 暂停最多 12 s
unsigned int APNEA_MS_FIXED     = 10000; // 若不随机，则固定 10 s

// 补偿性呼吸（暂停后通常出现更深/更大的呼吸）
unsigned int N_RECOVERY_BREATHS = 3;     // 补偿性呼吸个数
unsigned int INHALE_MS_RECO     = 1050;   // 补偿吸气更长
unsigned int EXHALE_MS_RECO     = 1050;   // 补偿呼气更长

// 周期重复设置
unsigned int N_AOP_CYCLES       = 100;   // AOP 周期次数（一次周期=正常呼吸若干 + 暂停 + 补偿呼吸）

// ====== 工具函数 ======
void pumpsStop() {
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE, 0);
  analogWrite(PIN_DEFLATE, 0);
}

void breatheOnce(unsigned int inhale_ms, unsigned int exhale_ms,
                 int speed_valve, int speed_deflate) {
  // Stage 1: 吸气（电磁阀开 + 充气泵）
  analogWrite(PIN_INFLATE, SPEED_SOLENOID);
  analogWrite(PIN_VALVE, speed_valve);
  analogWrite(PIN_DEFLATE, 0);
  delay(inhale_ms);

  // Stage 2: 呼气（电磁阀关 + 泄气泵）
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE, 0);
  analogWrite(PIN_DEFLATE, speed_deflate);
  delay(exhale_ms);

  pumpsStop(); // 小憩一下可选
}

void apneaPause(unsigned int pause_ms) {
  // Stage 3: 全停（平线段）
  pumpsStop();
  delay(pause_ms);
}

// ===== 主程序 =====
void setup() {
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE,   OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);

  Serial.begin(9600);
  Serial.println("=== AOP (Periodic Apnea) simulator start ===");

  // 可选：随机暂停需要随机种子
  randomSeed(analogRead(A0));

  for (unsigned int cyc = 1; cyc <= N_AOP_CYCLES; ++cyc) {
    Serial.print("\n[AOP Cycle "); Serial.print(cyc); Serial.println("]");

    // 1) 若干个正常小呼吸
    for (unsigned int k = 0; k < N_NORMAL_BEFORE_AP; ++k) {
      Serial.print("  Normal breath "); Serial.println(k + 1);
      breatheOnce(INHALE_MS_NORMAL, EXHALE_MS_NORMAL,
                  SPEED_VALVE_NORMAL, SPEED_DEFLATE_NORMAL);
    }

    // 2) 暂停（平线段）
    unsigned int pause_ms = RANDOM_APNEA
                            ? (unsigned int)random(APNEA_MS_MIN, APNEA_MS_MAX + 1)
                            : APNEA_MS_FIXED;
    Serial.print("  Apnea pause (ms): "); Serial.println(pause_ms);
    apneaPause(pause_ms);

    // 3) 补偿性大呼吸（更强/更长）
    for (unsigned int r = 0; r < N_RECOVERY_BREATHS; ++r) {
      Serial.print("  Recovery breath "); Serial.println(r + 1);
      breatheOnce(INHALE_MS_RECO, EXHALE_MS_RECO,
                  SPEED_VALVE_RECO, SPEED_DEFLATE_RECO);
    }

    // （可选）暂停后再接 1–2 个正常呼吸，回到平稳节律
    // breatheOnce(INHALE_MS_NORMAL, EXHALE_MS_NORMAL,
    //             SPEED_VALVE_NORMAL, SPEED_DEFLATE_NORMAL);
  }

  pumpsStop();
  Serial.println("\nAll AOP cycles completed.");
}

void loop() {
  // 空置
}

