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
            self.NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'NO_UNIQUE_PARALLEL_COMPONENT_MSG'
        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def __iter__(self):
        return self

    def next(self):
       self.idx += 1
       try:
           return Decimal(self.coordinates[self.idx-1])
       except IndexError:
           self.idx = 0
           raise StopIteration  # Done iterating.

    def __getitem__(self, i):
        if i < len(self.coordinates):
            return Decimal(self.coordinates[i])
        else:
            raise Exception("Vector __getitem__ out of critical value")

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

    def component_parallel_to(self, basis):
    	try:
    		u = basis.normalized()
    		weight = self.dot(u)
    		return u.times_scalar(weight)
    	except Exception as e:
    		if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
    			# There is no unique parallel component to the Zero Vector
    			raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
    	else:
    		raise e

    def component_orthogonal_to(self, basis):
    	try:
    		projection = self.component_parallel_to(basis)
    		return self.minus(projection)
    	except Exception as e:
    		if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
    			raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
    		else:
    			raise e

    def cross(self, v):
    	try:
    		x1, y1, z1 = self.coordinates
    		x2, y2, z2 = v.coordinates
    		new_coordinates = [y1*z2 - y2*z1, -(x1*z2 - x2*z1), x1*y2 - x2*y1]
    		return Vector(new_coordinates)
    	except Exception as e:
    		msg = str(e)
    		if msg == 'need more than 2 values to unpack':
    			self_embedded_in_R3 = Vector(self.coordinates + ('0',))
    			v_embedded_in_R3 = Vector(v.coordinates + ('0',))
    			return self_embedded_in_R3.cross(v_embedded_in_R3)
    		elif (msg == 'too many values to unpack' or msg == 'need more than 1 value to unpack'):
    			raise Exception(self.ONLY_DEFINE_IN_TWO_THREE_DIMS_MSG)
    		else:
    			raise e

    def area_of_parallelogram_with(self, v):
    	cross_product = self.cross(v)
    	return cross_product.magnitude()

    def area_of_triangle_with(self, v):
    	return self.area_of_parallelogram_with(v) / 2.0

'''

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

# 6. Parallel Component & Orthogonal Component
print ('\nParallel Component & Orthogonal Component')
v = Vector([3.039, 1.879])
w = Vector([0.825, 2.036])
print '#1'
# print v 
# print w
print v.component_parallel_to(w)

print '\n#2'
v = Vector([-9.88, -3.264, -8.159])
w = Vector([-2.155, -9.353, -9.473])
# print v 
# print w
print v. component_orthogonal_to(w)

print '\n#3'
v = Vector([3.009, -6.172, 3.692, -2.51])
w = Vector([6.404, -9.144, 2.759, 8.718])
# print v 
# print w
vpar = v.component_parallel_to(w)
vort = v.component_orthogonal_to(w)
print 'parallel component:', vpar
print 'orthogonal component:', vort

# 7. Cross Products
print '\nCross Products'
v = Vector(['8.462', '7.839', '-8.187'])
w = Vector(['6.984', '-5.975', '4.778'])
print '#1:', v.cross(w)

v = Vector(['-8.987', '-9.838', '5.031'])
w = Vector(['-4.268', '-1.861', '-8.866'])
print '#2:', v.area_of_parallelogram_with(w)

v = Vector(['1.5', '9.547', '3.691'])
w = Vector(['-6.007', '0.124', '5.772'])
print '#3:', v.area_of_triangle_with(w)

'''
