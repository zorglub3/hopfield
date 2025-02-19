import random

def pat(): return random.getrandbits(32)

patterns = []
for i in range(8):
    patterns.append(pat())

for i in range(10000):
    pattern = patterns[random.randrange(len(patterns))]
    print(pattern)

print("done") # This will stop the attractor network simulation, as it fails to parse an u32
