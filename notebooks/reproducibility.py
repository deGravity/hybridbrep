import random
def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)

def create_code_subset(codes, seed, size):
    indices = list(range(len(codes)))
    samples = create_subset(indices, seed, size)
    return codes[samples,:]