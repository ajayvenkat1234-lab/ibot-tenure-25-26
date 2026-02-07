int soundPin = 4;
int soundVal = LOW;
void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN , OUTPUT);
  pinMode(soundPin , INPUT);

}

void loop() {
  soundVal = digitalRead(soundPin);
  if(soundVal == HIGH){ 
    Serial.println("works");
    digitalWrite(LED_BUILTIN , HIGH);
    delay(2000);
    digitalWrite(LED_BUILTIN , LOW);

  }
  delay(50);

}
