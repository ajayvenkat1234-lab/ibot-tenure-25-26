#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>


#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void setup() {

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);

}

void loop() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(10,10);
  display.println("Hello World");
  display.drawRect(20, 20, 8, 5, WHITE);
  display.drawCircle(40,40,5,WHITE);
  display.display();
}
