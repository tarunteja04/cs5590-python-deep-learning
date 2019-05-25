word=input('Enter the string to be checked:')

currentLength = 1
longest = 1
last_seen = {word[0] : 0}
i = 1

while i < len(word):
    letter = word[i]
    if letter in last_seen:
        i = last_seen[letter] + 1
        last_seen.clear()
        longest = max(longest, currentLength)
        currentLength = 0
    else:
        last_seen[letter] = i
        currentLength += 1
        i += 1

longest = max(longest, currentLength)
print('The longest substring is:\n',word[:longest],'\nLength of sub String is:\n',longest)