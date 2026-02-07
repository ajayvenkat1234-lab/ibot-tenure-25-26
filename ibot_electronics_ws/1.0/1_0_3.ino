const int buttonPin=8;
const int ledPin=12;
bool prev_button_state=HIGH;
bool ledstate=LOW;

void setup()
{
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin,OUTPUT);
}

void loop()
{
  bool button_state = digitalRead(buttonPin);
  if (prev_button_state == HIGH && button_state == LOW){
    ledstate = !ledstate;
  }
  digitalWrite(ledPin,ledstate);
  prev_button_state = button_state;
  delay(100);
}