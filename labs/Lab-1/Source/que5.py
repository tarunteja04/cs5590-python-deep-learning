class Person:  #base class
    per_count=0
    def __init__(self,name,email): #default constructor
          self.name=name
          self.email=email
          Person.per_count+=1
    def DisplayDetails(self):
        print("Name: ",self.name,"Email: ",self.email)
    def DispCount(self):
        print("Total count of persons: ",Person.per_count)

class Employee(Person):#Inheritance Usage
    def __init__(self,Emp_name,email,Employee_phone): #default init constructor
        Person.__init__(self,Emp_name,email)
        self.Employee_phone = Employee_phone
        self.Employee_phone=Employee_phone
    def DisplayDetails(self):
        Person.DisplayDetails(self)
        print("Employee_phone: ",self.Employee_phone)

class Passenger(Employee): #inheritance
    def __init__(self,name,email,emp_phone,Emp_name,e_email,Employee_phone,time,date): #default Init constructor
        super().__init__(name,email,Employee_phone)   #usage of super call
        self.emp_phone=emp_phone
        self.name=name
        self.email=email
        self.ename=Emp_name
        self.eemail=e_email
        self.ephone=Employee_phone
        self.time=time
        self.date=date
    def DisplayDetails(self):
        Person.DisplayDetails(self)
        print("Emp phone:", self.emp_phone)
        print("The Airplane time is: ",self.time)
        print("The Reserved date is: ",self.date)


class Book():
    def __init__(self,per_name,Travel_date, travel_time): #default init constructor
        self.person_name=per_name
        self.travel=Travel_date
        self.time=travel_time
    def DisplayDetaails(self):
        print("Person_name: ",self.person_name,"Travel date: ",self.travel,"Travel time: ",self.time)

class Flight(Person,Book): #multiple Inheritance
    def __init__(self,name,email,per_name,blood_pressure,sugar_level): #default init constructor
        # super().__init__(name,email)
        Person.__init__(self,name,email)
        Book.__init__(self,per_name,blood_pressure, sugar_level)

    def DisplayDetails(self):
        Book.DisplayDetaails(self)
        Person.DisplayDetails(self)


#creation of instances for above classes

person1=Person('Lakshman','lakshman95@gmail.com')
person2=Person('Praneeth','praneeththota@gmail.com')
Employee1=Employee('Tarun','tarunkasturi@gmail.com','9162628454')
Employee2=Employee('Pavan','pavan123@gmail.com','916454585')
Book1=Book('Lakshman','15 march 2019','12 am')
Book2=Book('Praneeth','20 April 2019','2 pm')
print("##Persons Count##")
Person.DispCount(Person)

#Flight attending persons
Flight1=Flight('Lakshman','lakshman95@gmail.com','lakshman','15 march 2019', '12am')
Flight2=Flight('Praneeth','praneeththota@gmail.com','praneeth','20 April 2019','2 pm')

print("###Employee Details###")
Employee1.DisplayDetails();
Employee2.DisplayDetails();

print("##Person Details##")
person1.DisplayDetails();
person2.DisplayDetails();

print("##Book Details##")
Book1.DisplayDetaails();
Book2.DisplayDetaails();

print("#Flight attending person details")
Flight1.DisplayDetaails();


Passenger1=Passenger("brundha","brundha@gmail.com","9162625252","Tarun","tarunkasturi@gmail.com","12345","9 40 PM","16 JUNE")
print("passenger Appointments-1")
Passenger1.DisplayDetails()

Passenger2=Passenger("Yasritha","yasritha@gmail.com","990909876","pavan","pavan123@gmail.com","898976","8 00 AM","21 JUNE")
print("Passenger Appointments-2")
Passenger2.DisplayDetails()