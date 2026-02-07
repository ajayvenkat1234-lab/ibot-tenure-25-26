//EP25B004
//V.AJAY
int analogPin=A0;
void setup()
{
  Serial.begin(9600);
}

void loop()
{ 
  int ir = analogRead(analogPin);
  Serial.println(ir);
  delay(50);
}
