n=int(input("Total number of  plants?"))
x=(input("space between the plants height"))
sum=0.0
list=x.split()
for num in list:
    sum=sum+int(num)
print("Average height value is:",sum/n)


