import math
import random


def generate_uniform_circle(center, radius, num):
    output = []
    for _ in range(num):
        r = random.random() * radius
        theta = random.random() * 2 * math.pi
        output.append((center[0] + r * math.cos(theta), center[1] + r * math.sin(theta)))
    return output
        