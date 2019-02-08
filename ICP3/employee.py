class Employee():  #Taking the employee class(superclass)
    empCount = 0   #Initially making employee count to zero
    empSal = [];
    def __init__(self,name,family,salary,department): # constructor with attributes
        self.empname = name
        self.empfamily = family
        self.empsalary = salary
        self.empdepartment = department
        Employee.empCount +=1  #counts the number of employees
        Employee.empSal.append(salary)  #appends salaray attribute

    def avg_salary(self):             #method for caluculating avg salary
        print('the average salary is')
        sumSal = 0;
        for sal in Employee.empSal:
            sumSal = sumSal+ int(sal);       #adding all the salary
        return sumSal/len(Employee.empSal)   #caluculating average

class FulltimeEmployee(Employee):             #taking the fulltimeemployee class(subclass)
    def __init__(self,name,family,salary,department):
        Employee.__init__(self,name,family,salary,department)


emp1 = FulltimeEmployee('tarun',' teja','100','Computers');      #passing the attributes
emp2 = FulltimeEmployee('laxman','kumar','200','electronics');
emp3 = FulltimeEmployee('pavan','kumar','300','electrical');
print(FulltimeEmployee.empCount)  #inherits characteristics from parent class
print(FulltimeEmployee.empSal)    #printing the salary
avgSal = FulltimeEmployee.avg_salary(Employee);
print(avgSal)                     #printing the average salary