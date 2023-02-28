
'''
TASKS:
1. Find out the number of unique dialogue speakers in the sample conversation?
2. Create a new text file by the name of the dialogue speaker and store the unique words
spoken by that character in the respective text file. Make sure there is only one word
every line.
'''

import re
with open ("conv.txt", "r") as f:
    speaker = set ()
    for i in f.readlines ():
        name = re.search ("^([A-Z ]+): .+", i)
        if name:
            speaker.add (name.group(1))
    print (speaker)
    print ("Number of unique speakers in the conservation:", len(speaker))
    for name in speaker:
        f.seek (0)
        words = []
        for line in f.readlines ():
            if line.startswith (name):
                words.extend ((line.upper().replace(',','').replace('.','').replace('?','').replace('!','')).split ())
        with open ("{0}.txt".format(name), "w") as a:
            for i in words:
                if words.count (i) == 1:
                    a.write (i + "\n")