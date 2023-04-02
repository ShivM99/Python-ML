#Game with function
def position(): #Input the position user want to change
    try:
        ind = int(input("Enter the position from (1-5) you want to change: "))
        if ind not in range(1,6):
            print ("Input position out of range")
            return False
    except:
        print ("Your input is not an integer")
        return False
    return (ind)
def new(l,ind): #Alter the list for user-input
    new = input("Enter the new value you want to put: ")
    l[ind-1] = new
    return l
def more(): #Ask the user whether he want to play further
    x = input("Do you want to play ahead? (Y/N): ")
    if x == 'Y' or x == 'y':
        return True
    elif x == 'N' or x == 'n':
        return False
    else:
        print ("Invalid input. Please enter either 'Y' for yes or 'N' for no")
        return ("Invalid")
lt = [1,2,3,4,5]
print ("Current list:",lt)
ask = True
while ask:
    pos = position()
    if pos == False:
        continue
    else:
        print ("New list:",new(lt,pos))
    ask = more()
    while ask == "Invalid":
        ask = more()

