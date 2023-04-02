#Game
x = True
l = [1,2,3,4,5]
print ("Current list:",l)
while x:
    try:
        ind = int(input("Enter the position from (1-5) you want to change: "))
        if ind not in range(1,6):
            print ("Input position out of range")
            continue
    except:
        print ("Your input is not an integer")
        continue
    new = input("Enter the new value you want to put: ")
    l[ind-1] = new
    print ("New list:",l)
    while x:
        x = input("Do you want to play ahead? (Y/N): ")
        if x == 'Y' or x == 'y':
            break
        elif x == 'N' or x == 'n':
            x = False
        else:
            print ("Invalid input. Please enter either 'Y' for yes or 'N' for no")
            continue
