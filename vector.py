from math import sqrt

class Vector(object):

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    # Operating on Vectors
    def plus(self, v):
        new_coordinates = [x+y for x,
                           y in zip(self.coordinates, v.coordinates)]
        # new_coordinates = []
        # n = len(self.coordinates)
        # for i in range(n):
        # 	new_coordinates.append(self.coordinates[i] + v.coordinates[i])
        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [x-y for x,
                           y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [c*x for x in self.coordinates]
        return Vector(new_coordinates)

    # define Magnitude & Direction
    def magnitude(self):
    	coordinates_squared = [x**2 for x in self.coordinates]
    	return sqrt(sum(coordinates_squared))

    def normalized(self):
    	try:
    		magnitude = self.magnitude()
    		return self.times_scalar(1./magnitude)

    	except ZeroDivisionError:
    		raise Exception("Cannot normalize the zero Vector")

# 1. test
my_vector = Vector([1, 2, 3])
print my_vector

my_vector2 = Vector([2, 3, 4])
my_vector3 = my_vector2
print my_vector2 == my_vector3
print my_vector3 == my_vector
print my_vector2 == my_vector

# 2. Operating
v = Vector([8.218, -9.341])
w = Vector([-1.129, 2.111])
print v.plus(w)
print v.minus(w)
print v.times_scalar(2)

# 3. Magnitude & Direction
v = Vector([-0.221, 7.437])
print ("Magnitude(-0.221, 7.437) = " + str(v.magnitude()))

v = Vector([5.581, -2.136])
print ("Normalized([5.581, -2.136]) = " + str(v.normalized()))


