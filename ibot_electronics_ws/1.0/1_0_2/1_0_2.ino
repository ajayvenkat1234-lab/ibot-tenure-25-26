//EP25B004
//V.AJAY
int LED=9;
void setup()
{

  pinMode(LED, OUTPUT);
  Serial.begin(9600);
}
void breathe_in(){
  int brightness=0;
  while (brightness<=255){
  analogWrite(LED,brightness);
  Serial.println(brightness);
  brightness+=5;
  delay(10); // Wait for 10 millisecond(s)
}
}
void breathe_out(){
  int brightness=255;
  while( brightness>=0){
  analogWrite(LED,brightness);
      Serial.println(brightness);
  brightness-=5;
  delay(10); // Wait for 10 millisecond(s)
}
}
void loop()
{
  breathe_in();
  breathe_out();
}
