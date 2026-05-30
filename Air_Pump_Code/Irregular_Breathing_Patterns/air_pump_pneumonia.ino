// Define motor and solenoid valve control pins
const int PIN_INFLATE = 3;    
const int PIN_VALVE   = 4;    
const int PIN_DEFLATE = 5;    

// ----- Mode A: deeper breath -----
const int A_INFLATE_SPEED = 220;
const int A_DEFLATE_SPEED = 220;
const int A_INFLATE_TIME  = 800;  // ms
const int A_DEFLATE_TIME  = 800;  // ms

// ----- Mode B: shallow breath -----
const int B_INFLATE_SPEED = 200;
const int B_DEFLATE_SPEED = 200;
const int B_INFLATE_TIME  = 400;  // ms
const int B_DEFLATE_TIME  = 400;  // ms

void doBreath(int inflateSpeed, int deflateSpeed,
              int inflateTime, int deflateTime)
{
  // Inflate
  analogWrite(PIN_INFLATE, inflateSpeed);
  analogWrite(PIN_VALVE,   inflateSpeed);
  analogWrite(PIN_DEFLATE, 0);
  delay(inflateTime);

  // Deflate
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, deflateSpeed);
  delay(deflateTime);

  // Relax
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE,   0);
  analogWrite(PIN_DEFLATE, 0);
}

void setup() {
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE,   OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);
  Serial.begin(9600);

  for (int cycle = 0; cycle < 1000; cycle++) {

    if (cycle % 2 == 0) {
      Serial.println("Mode A: deeper breath");
      doBreath(A_INFLATE_SPEED, A_DEFLATE_SPEED,
               A_INFLATE_TIME, A_DEFLATE_TIME);
    }
    else {
      Serial.println("Mode B: shallow breath");
      doBreath(B_INFLATE_SPEED, B_DEFLATE_SPEED,
               B_INFLATE_TIME, B_DEFLATE_TIME);
    }
  }
}

void loop(){}
