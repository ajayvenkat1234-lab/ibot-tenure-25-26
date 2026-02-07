int ledPin = 8;
int pirPin = 4;
int pirVal = LOW;
void setup() {
  pinMode(ledPin,OUTPUT);
  pinMode(pirPin,INPUT);
  Serial.begin(9600);

}

void loop() {
  pirVal = digitalRead(pirPin);
  if (pirVal == HIGH){
    Serial.println("Motion Detected");
    digitalWrite(ledPin,HIGH);
  }
  else{
  Serial.println("No Motion");
  digitalWrite(ledPin,LOW);
  }
  delay(200);


}
