#SPY GAME: Write a function that takes in a list of integers and returns True if it contains 007 in order
def james_bond(l):
    try:
        c = l.copy()
    except:
        pass
    c.remove(0)
    if 0 in l and 0 in c and 7 in l:
        z1 = l.index(0)
        z2 = c.index(0)
        s = l.index(7)
        if z1<s and z2<s:
            return True
    return False
print ("Enter a non-integer to end user input")
lt = []
while True:
    try:
        x = int(input("Enter the number: "))
    except:
        break
    lt.append(x)
print (james_bond(lt))
