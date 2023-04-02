#Guessing game
from random import randint
print ("Enter a non-integer to end user input")
x = randint(1,100)
count = 0
while True:
    try:
        i = int(input("Enter an integer between 1 and 100: "))
        if i<1 or i>100:
            print ("OUT OF BOUNDS")
            continue
        if count == 0:
            pre = i
    except:
        break
    if i == x:
        print (f"You guessed correctly after {count} guesses")
        break
    elif abs(x-i) < abs(x-pre):
        print ("WARMER")
    elif abs(x-i) > abs(x-pre):
        print ("COLDER")
    elif abs(x-i) <= 10:
        print ("WARM")
    elif abs(x-i) > 10:
        print ("COLD")
    pre = i
    count += 1
