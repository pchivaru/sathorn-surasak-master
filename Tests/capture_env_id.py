import re

str = '123-0'

results = re.search(r'(\d+)-(\d+)', str)

print(results[1])
print(results[2])