int receiverPin = A0;
void setup() {
  Serial.begin(9600);

}

void loop() {
  int receiver = analogRead(receiverPin);
  Serial.println(receiver);
  delay(200);


}
