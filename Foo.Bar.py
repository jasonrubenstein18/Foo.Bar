"""
Formatting here isn't great, didn't anticipate loading to GitHub and can't go back to prompts on solved problems
"""

list = [1, 2, 3, 4, 4, 5, 6]
list2 = [1, 2, 3, 4, 4, 5, 6, 7, 8, 8]
my_list = list2


#1
# check dtype of data and remove if number appears more than n times
# labor shifts challenge
def solution(data, n):
    if type(data) == type([]):
        for i in data:
            if data.count(i) > n:
                data = [x for x in data if x != i]
            else:
                continue
    else:
        for i in data:
            if data.count(i) > n:
                data = [x for x in enumerate(data) if x != i]
            else:
                continue
    return data


solution(list, 1)
solution(list2, 1)



#2
"""
7 12 18 25
4 8 13 19
2 5 9 14
1 3 6 10

[n * (n-1) / 2] + 1
"""


def answer(x, y):
    coords = ((x+y-1) * (x+y-2))/2+x
    return str(coords)

# testing
import random
from random import randint
answer(randint(1,100000),randint(1,100000))

#3-1
# Finding number of nodes in tree
"""
      7
  3       6
1   2   4   5
"""


def find_key(key, current_node, difference):
    right_node = current_node-1
    left_node = right_node-difference//2

    if right_node == key or left_node == key:
        return current_node
    else:
        if key <= left_node:
            return find_key(key, left_node, difference//2)
        else:
            return find_key(key, right_node, difference//2)


def solution(h, q):
    flux_converters = []
    keys = (2 ** h) - 1
    if h > 30 or h < 1:
        print('Out Of Scope Error; Height')
    if len(q) > 10000 or len(q) < 1 :
        print('Out Of Scope Error; Converter List')

    for i in range(len(q)):
        if q[i]<keys and q[i]>0:
            flux_converters.append(find_key(q[i], keys, keys-1))
        else:
            flux_converters.append(-1)
    return flux_converters


answer(3, [7,3,5,1])


# 2-2
def solution(n):
    try:
        float(n)
    except ValueError:
        pass
    else:
        if n.isdigit():
            n = int(n)
            result = 0
            while n > 1:
                if n % 2 == 0:
                    n = n / 2
                else:
                    n = (n - 1) \
                        if n == 3 or n % 4 == 1 \
                        else n + 1
                result += 1
            return result


solution("15.23")

solution("15")


n = "124"
n = int(n)

#3-2
# finding terminal state of markov chain
#import numpy as np
from fractions import Fraction
from fractions import gcd
from functools import reduce


def matrix_work(A):
    n = len(A)  # The input matrix is squared

    # empty table
    table = [[0]*2*n for i in range(n)]

    # Copy initial
    for i in range(n):
        for j in range(n):
            table[i][j] = A[i][j]
        table[i][i+n] = 1  # Diagonal matrix

    # Pivot each row
    for i in range(n):
        # Normalize starting row
        scalar = table[i][i]
        for j in range(2*n):
            table[i][j] /= scalar

        # Subtract row to all next and prior
        for j in range(n):
            if j != i:
                scalar = table[j][i]
                for k in range(2*n):
                    table[j][k] -= scalar * table[i][k]

    B = [table[i][n:] for i in range(n)]
    return B


def get_transition_matrix(m):
    n = len(m)

    P = [[0]*n for _ in range(n)]  # Initialize matrix

    map_states_tra = {}  # Transitional states
    map_states_abs = {}  # Finals states

    for i in range(n):
        total_weights = sum(m[i])
        if total_weights == 0:
            map_states_abs[i] = len(map_states_abs)

            pos = n - len(map_states_abs)
            P[pos][pos] = 1
        else:
            map_states_tra[i] = len(map_states_tra)

            for j in range(n):  # Normalize weights
                m[i][j] = Fraction(m[i][j], total_weights)

    len_tran = len(map_states_tra)
    for k, v in map_states_abs.items():
        map_states_abs[k] += len_tran

    map_states_tra.update(map_states_abs)
    map_states = map_states_tra

    inv_map_states = {v: k for k, v in map_states.items()}
    # rewrite with right IDs
    for i in range(len_tran):
        for j in range(n):
            P[i][j] = m[inv_map_states[i]][inv_map_states[j]]

    Q = [x[0:len_tran] for x in P[0:len_tran]]
    R = [x[len_tran:] for x in P[0:len_tran]]

    return Q, R


def lcm(a, b):
    return a * b // gcd(a, b)


def normalize_probs(probs):
    lcm_glob = reduce(lcm, [fract.denominator for fract in probs])

    normalized_probs = [fract.numerator * lcm_glob // fract.denominator for fract in probs]
    normalized_probs.append(lcm_glob)

    return normalized_probs


def solution(m):
    # The problem is equivalent to computing the absorbing markov chain
    # Case where the initial state is final (all state finals)
    if sum(m[0]) == 0:
        return [1, 1]

    # Solve
    Q, R = get_transition_matrix(m)

    for i in range(len(Q)):
        Q[i][i] -= 1
    F = matrix_work(Q)

    probs = []
    for i in range(len(R[0])):
        cell = 0
        for j in range(len(F)):
            cell += -F[0][j]*R[j][i]  # correct negative sign here
        probs.append(cell)

    probs_normalized = normalize_probs(probs)
    return probs_normalized

solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]])


def backprop(v1, i1, v2, i2):
    # back propagate distribution
    lenV = len(v1)
    indices = (set(range(lenV)) - {i1, i2})
    sum2 = sum(v2)
    output = [0 for i in v1]
    for i in indices:
        output[i] = sum2 * v1[i] + v1[i2] * v2[i]
    gc = gcd_list(output)
    output = [int(i / gc) for i in output]
    return output

def gcd(a, b):
    if (b == 0):
        return a
    else:
        return gcd(b, a % b)

def gcd_list(list):
    L = len(list)
    out = 0
    for i in range(0, L):
        out = gcd(out, list[i])
    return out

def solution(m):
    height = len(m)
    width = len(m[0])
    matrix = list(m)
    for i, element in enumerate(matrix):
        element[i] = 0
    sums = [sum(i) for i in matrix]
    terms = [i for i, item in enumerate(sums) if item == 0]
    not_terms = list((set(range(height)) - set(terms)))
    length = len(not_terms)

    for i in range(0, length - 1):
        indB = not_terms[length - i - 1]
        for j in range(0, length - 1):
            indA = not_terms[j]
            matrix[indA] = backprop(matrix[indA], indA, matrix[indB], indB)
    output = []
    for i in terms:
        output.append(matrix[0][i])
    total = sum(output)
    output.append(total)
    if total == 0:
        output = [1 for i in terms]
        output.append(len(terms))
    return output


print(solution([[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))

# 3-3
import time

maze = ([[0, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
from collections import deque


class Node:

    def __init__(self, x, y, power, grid):
        self.x = x
        self.y = y
        self.power = power
        self.grid = grid

    def __hash__(self):
        return self.x ^ self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.power == other.power

    def get_adjacents(self):
        adjacents = []
        x = self.x
        y = self.y
        power = self.power
        grid = self.grid
        rows = len(grid)
        columns = len(grid[0])

        if x > 0:
            wall = grid[y][x - 1] == 1
            if wall:
                if power > 0:  # if still has ability to break wall on iteration
                    adjacents.append(Node(x - 1, y, power - 1, grid))
            else:
                adjacents.append(Node(x - 1, y, power, grid))

        if x < columns - 1:
            wall = grid[y][x + 1] == 1
            if wall:
                if power > 0:
                    adjacents.append(Node(x + 1, y, power - 1, grid))
            else:
                adjacents.append(Node(x + 1, y, power, grid))

        if y > 0:
            wall = grid[y - 1][x] == 1
            if wall:
                if power > 0:
                    adjacents.append(Node(x, y - 1, power - 1, grid))
            else:
                adjacents.append(Node(x, y - 1, power, grid))

        if y < rows - 1:
            wall = grid[y + 1][x]
            if wall:
                if power > 0:
                    adjacents.append(Node(x, y + 1, power - 1, grid))
            else:
                adjacents.append(Node(x, y + 1, power, grid))

        return adjacents


class BunnyEscapeRoute:

    def __init__(self, grid, power):
        self.grid = grid
        self.rows = len(grid)
        self.columns = len(grid[0])
        self.power = power

    def get_escape_route_length(self):
        source = Node(0, 0, self.power, self.grid)
        queue = deque([source])
        distance_map = {source: 1}

        while queue:
            current_node = queue.popleft()

            if current_node.x == self.columns - 1 and \
                    current_node.y == self.rows - 1:
                return distance_map[current_node]

            for child_node in current_node.get_adjacents():
                if child_node not in distance_map.keys():
                    distance_map[child_node] = distance_map[current_node] + 1
                    queue.append(child_node)

        return 0  # Cannot escape, bunnies are doomed :(


def solution(map):
    route = BunnyEscapeRoute(map, 1)
    route_len = route.get_escape_route_length()
    return route_len

solution(maze)

