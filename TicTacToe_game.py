#Tic - tac - toe game
game = " 1 | 2 | 3 \n___|___|___\n 4 | 5 | 6 \n___|___|___\n 7 | 8 | 9 \n   |   |   "
print (game)
l = ['X','0']
y = input ("Do you want 'X' or '0'? You will play first.: ")
l.remove(y)
m = l.pop()
print ("You: ",y)
print ("Me: ",m)
while True:
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    i = 1
    while i < 10:
        try:
            n = int (input (f"Enter a position number from {l}: "))
            if n in l:
                l.remove (n)
            else:
                print ("Input not in range")
                continue
        except:
            print ("Invalid input")
            continue
        if i%2 != 0:
            game = game.replace (str(n), y)
            print (game)
            if (f" {y} | {y} | {y} " in game) or (f" {y} | 2 | 3 \n___|___|___\n {y} | 5 | 6 \n___|___|___\n {y} | 8 | 9 " in game) or (f" 1 | {y} | 3 \n___|___|___\n 4 | {y} | 6 \n___|___|___\n 7 | {y} | 9 " in game) or (f" 1 | 2 | {y} \n___|___|___\n 4 | 5 | {y} \n___|___|___\n 7 | 8 | {y} " in game) or (f" {y} | 2 | 3 \n___|___|___\n 4 | {y} | 6 \n___|___|___\n 7 | 8 | {y} " in game) or (f" 1 | 2 | {y} \n___|___|___\n 4 | {y} | 6 \n___|___|___\n {y} | 8 | 9 " in game):
                print ("YOU WON!!!")
                break
        else:
            game = game.replace (str(n), m)
            print (game)
            if (f" {m} | {m} | {m} " in game) or (f" {m} | 2 | 3 \n___|___|___\n {m} | 5 | 6 \n___|___|___\n {m} | 8 | 9 " in game) or (f" 1 | {m} | 3 \n___|___|___\n 4 | {m} | 6 \n___|___|___\n 7 | {m} | 9 " in game) or (f" 1 | 2 | {m} \n___|___|___\n 4 | 5 | {m} \n___|___|___\n 7 | 8 | {m} " in game) or (f" {m} | 2 | 3 \n___|___|___\n 4 | {m} | 6 \n___|___|___\n 7 | 8 | {m} " in game) or (f" 1 | 2 | {m} \n___|___|___\n 4 | {m} | 6 \n___|___|___\n {m} | 8 | 9 " in game):
                print ("I WON!!!")
                break
        i += 1
    else:
        print ("DRAW")
    y_n = input ("Do you want to play again? (Y/N): ").upper()
    if y_n == 'Y':
        i = 1
        continue
    elif y_n == 'N':
        break
    else:
        print ("Invalid input")
        break
