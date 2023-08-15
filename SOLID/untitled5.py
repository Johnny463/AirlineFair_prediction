from abc import ABC, abstractmethod

# Single Responsibility Principle (SRP)
class FileManager:
    def read_file(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def write_file(self, filename, content):
        with open(filename, 'w') as file:
            file.write(content)

    # Violation of SRP: Mixing responsibilities in the same class
    # def manage_directories(self):
    #     pass

    # Violation of SRP: Mixing responsibilities in the same class
    # def database_connection(self):
    #     pass


# Open/Closed Principle (OCP)
class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

# Violation of OCP: If new animal types are added, this class needs modification
# class Octopus(Animal):
#     def make_sound(self):
#         return "Squishy sound!"


class Lion(Animal):
    def make_sound(self):
        return "Roar!"


class Elephant(Animal):
    def make_sound(self):
        return "Trumpet!"


class Enclosure:
    def __init__(self, capacity):
        self.capacity = capacity
        self.animals = []

    def add_animal(self, animal):
        if len(self.animals) < self.capacity:
            self.animals.append(animal)
            return True
        return False


# Liskov Substitution Principle (LSP)
class ZooVisitor:
    def observe_animal(self, animal):
        animal_sound = animal.make_sound()
        print(f"The animal makes a sound: {animal_sound}")

# Violation of LSP: Octopus is not a proper substitute for Animal (contrived example)
# class Octopus(Animal):
#     def make_sound(self):
#         return "Squishy sound!"

#     # LSP violation: Octopus doesn't follow the same behavior as other animals
#     def swim(self):
#         return "Swimming gracefully!"


# Define a separate class for aquatic animals that can swim
class Octopus(Animal):
    def make_sound(self):
        return "Squishy sound!"

class Swimmer(ABC):
    @abstractmethod
    def swim(self):
        pass

class SwimmerOctopus(Octopus, Swimmer):
    def swim(self):
        return "Swimming gracefully!"

# Interface Segregation Principle (ISP)
class Feedable(ABC):
    @abstractmethod
    def feed(self):
        pass


class FeedableLion(Lion, Feedable):
    def feed(self):
        print("Feeding the lion...")

# Violation of ISP: Animals that don't need feeding are forced to implement feed()
# class Snake(Animal, Feedable):
#     def make_sound(self):
#         return "Hiss!"
#
#     def feed(self):
#         print("Feeding the snake...")



# Dependency Inversion Principle (DIP)
class Employee(ABC):
    @abstractmethod
    def perform_duty(self):
        pass


class ZooKeeper(Employee):
    def perform_duty(self):
        print("ZooKeeper is performing duties.")


class Veterinarian(Employee):
    def perform_duty(self):
        print("Veterinarian is performing duties.")


class Manager(Employee):
    def perform_duty(self):
        print("Manager is performing duties.")




class Zoo:
    def __init__(self, zoo_keeper, veterinarian, manager):
        self.zoo_keeper = zoo_keeper
        self.veterinarian = veterinarian
        self.manager = manager

    def start_operations(self):
        self.zoo_keeper.perform_duty()
        self.veterinarian.perform_duty()
        self.manager.perform_duty()


if __name__ == "__main__":
    # Single Responsibility Principle (SRP)
    file_manager = FileManager()
    file_manager.write_file("zoo.txt", "Welcome to the Zoo!")
    content = file_manager.read_file("zoo.txt")
    print("File content:", content)

    # Open/Closed Principle (OCP)
    lion = Lion()
    elephant = Elephant()
    enclosure = Enclosure(capacity=2)
    enclosure.add_animal(lion)
    enclosure.add_animal(elephant)

    # Liskov Substitution Principle (LSP)
    visitor = ZooVisitor()
    visitor.observe_animal(lion)
    swimmer_octopus = SwimmerOctopus()
    visitor.observe_animal(swimmer_octopus)
    # Interface Segregation Principle (ISP)
    lion = FeedableLion()
    lion.feed()

    # Dependency Inversion Principle (DIP)
    zoo_keeper = ZooKeeper()
    veterinarian = Veterinarian()
    manager = Manager()
    # Instead of directly instantiating and managing employees, use a list
    employees = [zoo_keeper, veterinarian, manager]
    zoo = Zoo(employees)
    zoo.start_operations()
