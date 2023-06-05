import re
 
# initializing string
test_str = '"'
 
# printing original String
print("The original string is : " + str(test_str))
 
# using sub() to perform substitutions
# ord() for conversion.
res = (re.sub('.', lambda x: r'\u % 04X' % ord(x.group()), test_str))
 
# printing result
print("The unicode converted String : " + str(res))