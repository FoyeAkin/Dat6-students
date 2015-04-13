"""
This is an intro to regular expressions

I use 
https://regex101.com/#python 
to check my work!
"""


import re

# flow:
# create a re pattern object
# search (or match) it against text
# orgnize the captures patterns in groups

# \d matches a number
text = "Hello! My name is Sinan. It is 2014 and it's amazing."
pattern1 = re.compile("\d")
re.search(pattern1, text) # == a search object
# use group to get each instance in the regular expression
# \d is just ONE number, so it only finds the "2" in "2014"
re.search(pattern1, text).group(0)


# adding a + means "at least one" but potentially more
pattern2 = re.compile("\d+")
re.search(pattern2, text).group(0) # == '2014'

# use square brackets [] to match one of the items present
alphabet = 'abcfgfcgfgccffgfgcfcgfgfIBETITENDSHEREfg'
pattern3 = re.compile('cgf')
re.search(pattern3, alphabet).group(0)

phone_pattern = re.compile("\d\d\d-\d\d\d-\d\d\d\d")
# this will capture something like a phone pattern
re.search(phone_pattern, "my phone number is 609-4699992-6706 dude").group(0)


# . matches ANYTHING
all_of_the_text = "dmzhvbekuhvbc     dfljghwco87rc6geinsr6t4gi7rgwefiuvbekuhvbdfljghwco87rc6geinsr6t4gi7rgwefiu ywgsfybcstzvgbrtybte"
anything_pattern = re.compile(".+")
re.search(anything_pattern, all_of_the_text).group(0)



# \w matches any word character, alphanumeric
more_text = "To be or not to be in 2015"
text_pattern = re.compile('\w*')
re.search(text_pattern, more_text).group(0)
# notice how it stopped at the space, because a space is NOT
# alphanumeric!

sentence = "I am happy to be here"
# will match exactly "am" and then anything, and then "to"
complicated_pattern = re.compile("am .* to")
re.search(complicated_pattern, sentence).group(0)

# if you want to match an actual period, do \.
# try making an email pattern that will take in any numberof alphanumeric 
# characters and periods
# accompanied with an @ and the ending pattern we will assume is just .com
email_pattern = re.compile("[\w\.]*@[\w\.]+\.com")
email_re = re.search(email_pattern, "my email address is sinan.u.ozdemir@gmail.com")
email_re.group(0) # == 'sinan.u.ozdemir@gmail.com'
