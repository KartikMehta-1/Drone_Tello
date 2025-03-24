#pygame Used to get commands from keyboard. It is used for creating games
import pygame 

#A window in which keypresses will be detected
def init():
    pygame.init()
    win = pygame.display.set_mode((400,400))

#Function to get keypresses. If we give it a key name, it will tell us whether it is pressed or not
def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame,'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

#test main function
def main():
    if getKey("Left"):
        print("Left pressed")
    
    
if __name__ == '__main__':
    init()
    while True:
        main()
