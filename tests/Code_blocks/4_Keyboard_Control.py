from djitellopy import tello
from time import sleep
import KeyPressModule as kp


kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getKeyboardInput():
    lr,fb,ud,yv = 0,0,0,0
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): ud = speed
    elif kp.getKey("DOWN"): ud = -speed

    if kp.getKey("w"): fb = speed
    elif kp.getKey("s"): fb = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()
    return [lr,fb,ud,yv]


me.takeoff()

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])