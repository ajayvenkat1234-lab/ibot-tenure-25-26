#include <Servo.h>
Servo servo;
void increase(){
  for (int servoval = 0; servoval<=180; servoval++){
    servo.write(servoval);
    delay(15);
  }
}
void decrease(){
  for (int servoval = 180; servoval>=0; servoval--){
    servo.write(servoval);
    delay(15);
  }
}
void setup() {
  servo.attach(9);

}

void loop() {

  increase();
  decrease();


}
