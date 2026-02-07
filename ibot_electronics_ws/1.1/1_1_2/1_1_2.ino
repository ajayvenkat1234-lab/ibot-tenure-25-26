//EP25B004
//V.AJAY
int analogPin=A0;
void setup()
{
  Serial.begin(9600);
}

void loop()
{ 
  int ldr = analogRead(analogPin);
  Serial.println(ldr);
  delay(50);
}
