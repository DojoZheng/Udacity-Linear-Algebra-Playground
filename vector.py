from math import sqrt, acos, pi
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(self.coordinates)
            self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'CANNOT_NORMALIZE_ZERO_VECTOR_MSG'

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
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

    # define Magnitude & Direction
    def magnitude(self):
    	coordinates_squared = [x**2 for x in self.coordinates]
    	return sqrt(sum(coordinates_squared))

    def normalized(self):
    	try:
    		# We have to use Decimal in order to avoid boundary error
    		magnitude = self.magnitude()
    		return self.times_scalar(Decimal('1.0')/Decimal(magnitude))
    	except ZeroDivisionError:
    		raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def dot(self, v):
    	return sum([x*y for x,y in zip(self.coordinates, v.coordinates)])

    def angle_with(self, v, in_degrees = False):
    	try:
    		u1 = self.normalized()
    		u2 = v.normalized()
    		angle_in_radians = acos(u1.dot(u2))

    		if in_degrees:
    			degrees_per_radian = 180. / pi
    			return angle_in_radians * degrees_per_radian
    		else:
    			return angle_in_radians

    	except Exception as e:
    		if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
    			raise Exception("Cannot compute an angle with the zero vector")
    	else:
    		raise e

    def is_zero(self, tolerance=1e-10):
    	return self.magnitude() < tolerance

    def is_orthogonal_to(self, v, tolerance=1e-10):
    	return abs(self.dot(v)) < tolerance

    def is_parallel_to(self, v, tolerance=1e-10):
    	return (self.is_zero() or
    			v.is_zero() or
                abs(self.angle_with(v) - 0) < tolerance or
                abs(self.angle_with(v) - pi) < tolerance)

# 1. test
my_vector = Vector([1, 2, 3])
print my_vector

my_vector2 = Vector([2, 3, 4])
my_vector3 = my_vector2
print my_vector2 == my_vector3
print my_vector3 == my_vector
print my_vector2 == my_vector
print('\n')

# 2. Operating
v = Vector([8.218, -9.341])
w = Vector([-1.129, 2.111])
print v.plus(w)
print v.minus(w)
print v.times_scalar(2)
print('\n')

# 3. Magnitude & Direction
v = Vector(['-0.221', '7.437'])
print ("Magnitude(-0.221, 7.437) = " + str(v.magnitude()))

v = Vector(['5.581', '-2.136'])
print ("Normalized([5.581, -2.136]) = " + str(v.normalized()))
print('\n')

# 4. Inner/ Dot Product & Angle
v = Vector(['7.887', '4.138'])
w = Vector(['-8.802', '6.776'])
print("v.w = " + str(v.dot(w)))

v = Vector(['3.183', '-7.627'])
w = Vector(['-2.668', '5.319'])
print ("v angle with w:" + str(v.angle_with(w)))
print('\n')

# 5. Parallel & Orthogonal
# v = Vector(['-7.579', '-7.88'])
# w = Vector(['22.737', '23.64'])
v = Vector(['1', '0'])
w = Vector(['1', '0'])
print v
print w
print 'is parallel :', v.is_parallel_to(w)
print 'is orthogonal :', v.is_orthogonal_to(w)

