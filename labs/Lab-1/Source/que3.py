P = input("enter names of students in python class:")
Python = set(P.split(" "))# input get sepearted by spaces between them
#print(Python)

# list of students who took web
W = input("enter names of students in Web class:")
web = set(W.split(" "))
#print(Web)


# printing the common students who took both python and web using & operator
print("printing the common students who took both python and web")
print(Python & web)

# storing the list of students who took only python to the variable 'a'
a = Python-web

# storing the list of students who took only web to the variable 'a'
b = web-Python

# printing the list of students who are not in common in both python and web
print("printing the list of students who are not in common in both python and web:")
print(a.union(b))