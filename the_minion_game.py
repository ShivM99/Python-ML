x = input ("Enter a string: ")
l = list(letter for letter in x.upper())
vow = ['A','E','I','O','U']
l_con = []
c_con = 0
l_vow = []
c_vow = 0
for i in range(len(l)):
    if l[i] not in vow:
        for j in range(i+1,len(l)+1):
            l_con.append(''.join(l[i:j]))
            c_con += 1
    elif l[i] in vow:
        for j in range(i+1,len(l)+1):
            l_vow.append(''.join(l[i:j]))
            c_vow += 1
print ("Consonant list:",l_con,"Consonant count:",c_con)
print ("Vowel list:",l_vow,"Vowel count:",c_vow)
