def Convert(tup, di): #Convert a list of Tuples into Dictionary
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


# Driver Code
tups = [( 'John', ('Physics', 80)),('Daniel', ('Science',90)),('John', ('Science', 95)),
        ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
dictionary = {}#for sorting of given tuple
print(Convert(tups, dictionary))
