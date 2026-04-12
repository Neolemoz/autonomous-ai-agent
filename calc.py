def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return 'Error: Division by zero'
    return x / y

if __name__ == "__main__":
    print('Addition: 3 + 5 =', add(3, 5))
    print('Subtraction: 10 - 4 =', subtract(10, 4))
    print('Multiplication: 6 * 7 =', multiply(6, 7))
    print('Division: 8 / 2 =', divide(8, 2))
