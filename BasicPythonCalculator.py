#function to calculate
def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def multiply(x,y):
    return x * y

def divide(x,y):
    if y == 0:
        return "Error"
    else:
        return x / y

def calculator():
    print("Select Option")
    print("1.add")
    print("2.substract")
    print("3.multiply")
    print("4.divide")

    while True:
        choice=input("Enter choice (1,2,3,4) :")
        if choice in ['1','2','3','4']:
            num1=float(input("Enter the first number"))
            num2=float(input("Enter the second number"))

            if choice=='1':
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice=='2':
                print(f"{num1} - {num2} = {substract(num1, num2)}")
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == '4':
                print(f"{num1} / {num2} = {divide(num1, num2)}")
        else:
            print("Invalid choice")

        next_calculation= input("Do you want to perform another calculation? (yes/no)")
        if next_calculation.lower() != "yes":
           break

    print("Exiting calculator.")

calculator()







