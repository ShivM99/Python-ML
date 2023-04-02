#Shuffle game
from random import shuffle
cups = [' ',' ',' ','O',' ']
shuffle(cups)
while True:
    try:
        guess = int(input("Enter your guess 1,2,3,4 or 5: "))
        break
    except:
        print ("Invalid input")
if cups[guess-1] == 'O':
    print ("Hoorah!!!")
else:
    print ("Oops!!! Better luck next time")
    print (f"'O' is present in the cup: {cups.index('O')+1}")
    print (cups)
