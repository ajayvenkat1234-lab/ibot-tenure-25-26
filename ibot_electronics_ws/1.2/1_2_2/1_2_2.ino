#include <LiquidCrystal.h>
LiquidCrystal mylcd(7, 8, 9, 10, 11, 12);
void setup() {
  mylcd.begin(16,2);

}

void loop() {
  mylcd.setCursor(0,0);
  mylcd.print("Hello World");
  mylcd.setCursor(0,1);
  mylcd.print("Ajay");
    
}
