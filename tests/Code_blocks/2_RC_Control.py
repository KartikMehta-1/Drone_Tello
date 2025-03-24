from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()

print(me.get_battery())

me.takeoff()
#send_rc_control(self, left_right_velocity: int, forward_backward_velocity: int, up_down_velocity: int, yaw_velocity: int):
me.send_rc_control(50,0,0,0) #move right by 50 cm

sleep(2)

me.send_rc_control(-50,0,0,0) #move right by 50 cm

sleep(2)

me.send_rc_control(0,0,0,0)

me.land()