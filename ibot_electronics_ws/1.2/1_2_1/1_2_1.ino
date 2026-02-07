int buzzerPin = 8;
void setup() {
  

}

void loop() {
  tone(buzzerPin,500);
  delay(1000);
  noTone(buzzerPin);
  delay(1000);

}
