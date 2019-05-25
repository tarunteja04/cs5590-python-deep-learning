# computes net bank amount based on the input
# "D" for deposit, "W" for withdrawal

# define a variable for main amount
remaining_amount = 0

while True:
	# input the transaction
	str = input ("Enter transaction: ")

	# get the value type and amount to the list
	# seprated by space
	transaction = str.split(" ")

	# get the value of transaction type and amount
	# in the separated variables
	type = transaction[0]
	amount = int(transaction[1])

	if type=="Deposit" or type=="deposit":
		remaining_amount += amount
	elif type=="Withdraw" or type=="withdraw":
		remaining_amount -= amount
	else:
		pass

	#input choice
	str = input ("want to continue (Y for yes) : ")
	if not (str[0] =="Y" or str[0] =="y") :
		# break the loop
		break

# print the remaining amount
print("Remaining amount: ", remaining_amount)