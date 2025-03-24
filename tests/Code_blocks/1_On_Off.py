from djitellopy import Tello
import logging
import time

Tello.LOGGER.setLevel(logging.DEBUG)

tello = Tello()

tello.connect()
battery = tello.query_battery()
print(battery)
#tello.takeoff()
print(tello.get_battery())

#tello.move_left(100)
#tello.rotate_counter_clockwise(90)
#tello.move_forward(100)o
#tello.land()
