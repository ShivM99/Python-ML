'''
Julius Caesar protected his confidential information by encrypting it using a cipher. Caesar's cipher shifts each letter by a number of letters. If the shift takes you past the end of the alphabet, just rotate back to the front of the alphabet. In the case of a rotation by 3, w, x, y and z would map to z, a, b and c.
Original alphabet:      abcdefghijklmnopqrstuvwxyz
Alphabet rotated +3:    defghijklmnopqrstuvwxyzabc
'''
import string
s = input ("Enter the string: ")
k = int (input ("Enter the number of rotation: "))
s = list(s)
upp = list(string.ascii_uppercase) * 100
low = list(string.ascii_lowercase) * 100
for i in range (len(s)):
    if s[i] in upp:
        s[i] = upp[upp.index(s[i])+k]
    elif s[i] in low:
        s[i] = low[low.index(s[i])+k]
print (''.join(s))
