// Define motor and solenoid valve control pins
const int PIN_INFLATE = 3;    // Solenoid valve
const int PIN_VALVE = 4;      // Inflation pump
const int PIN_DEFLATE = 5;    // Deflation pump
//======================================================

// Define duty cycle values (0-255)
const int INFLATE_SPEED = 255;  // Solenoid valve fully open (Note: solenoid valve speed cannot be adjusted, do not change the duty cycle)
const int VALVE_SPEED = 255;    // Inflation pump at full speed (to adjust pump airflow, change 255 to a value between 0-255; larger numbers mean faster speed)
const int DEFLATE_SPEED = 255;  // Deflation pump speed (adjustment same as above)
//======================================================

// Main program starts
void setup() {
  // Initialize motor control pins as output mode
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE, OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);
  
  // Start serial communication for debugging
  Serial.begin(9600);
  Serial.println("Motor control system started");
  Serial.print("Inflation pump speed: ");
  Serial.print((INFLATE_SPEED / 255.0) * 100);
  Serial.println("%");
  Serial.print("Solenoid valve speed: ");
  Serial.print((VALVE_SPEED / 255.0) * 100);
  Serial.println("%");
  Serial.print("Deflation pump speed: ");
  Serial.print((DEFLATE_SPEED / 255.0) * 100);
  Serial.println("%");
//==================================================================
  //residual volume
  analogWrite(PIN_INFLATE, INFLATE_SPEED);  // Solenoid valve fully open
  analogWrite(PIN_VALVE, VALVE_SPEED);      // Inflation pump runs at set speed
  analogWrite(PIN_DEFLATE, 0);              // Deflation pump stopped
  delay(1250);                             

  // Work cycle
  for (int cycle = 0; cycle < 1000; cycle++) {       
  // The number of cycles can be customized by changing the number in (cycle < 3) above; for example, for 10 cycles, use (cycle < 10)
    Serial.print("Starting cycle ");
    Serial.print(cycle + 1);
    Serial.println();
 //====================================================================   
    // Stage 1: Inflate
    Serial.println("Inflation stage start");
    analogWrite(PIN_INFLATE, INFLATE_SPEED);  // Solenoid valve fully open
    analogWrite(PIN_VALVE, VALVE_SPEED);      // Inflation pump runs at set speed
    analogWrite(PIN_DEFLATE, 0);              // Deflation pump stopped
    delay(600);                              // Normal Breathing pattern: 40 BPM, 31.25 mL (flow rate = 2500mL/min)
    
    // Stage 2: Deflate
    Serial.println("Deflation stage start");
    analogWrite(PIN_INFLATE, 0);              // Solenoid valve closed
    analogWrite(PIN_VALVE, 0);                // Inflation pump stopped
    analogWrite(PIN_DEFLATE, DEFLATE_SPEED);  // Deflation pump runs at set speed
    delay(550);                              // Normal Breathing pattern: 40 BPM, 31.25 mL

    // Stage 3: Stop everything
    Serial.println("Stop stage start");
    analogWrite(PIN_INFLATE, 0);              // Inflation pump stopped
    analogWrite(PIN_VALVE, 0);                // Solenoid valve closed
    analogWrite(PIN_DEFLATE, 0);               // Deflation pump stopped
  }
  
  Serial.println("All cycles completed");
}

void loop() {
  // Main loop is empty; program stops after executing setup
}
