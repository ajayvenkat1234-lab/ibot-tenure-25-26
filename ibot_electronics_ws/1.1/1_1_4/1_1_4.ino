//EP25B004
//V.AJAY
float duration;
float distance;
int trigpin=3;
int echopin=2;
void setup()
{
  pinMode(trigpin,OUTPUT);
  pinMode(echopin,INPUT);
  Serial.begin(9600);
}

void loop()
{
  digitalWrite(trigpin,LOW);
  delayMicroseconds(2);
  digitalWrite(trigpin,HIGH);
  delayMicroseconds(10);
  digitalWrite(trigpin,LOW);
  duration=pulseIn(echopin,HIGH,30000);
  distance=( duration*0.0343 )/2;	//distance in cm
  
  if (duration==0){
    Serial.println("No Object Found");
    
  }
  else{
    Serial.println("Dist:");
    Serial.println(distance);
    Serial.println("cm    ");
  }
  delay(300);
    
}