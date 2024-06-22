import struct
import random
import socket
import os
import numpy as np

def generate_ipv4_data(filename_prefix, file_num, num_entries, s, reservoir_size):
    filename = f"{filename_prefix}{file_num}.dat"
    reservoir = []
    for i in range(num_entries):
        srcIP = socket.inet_aton(".".join(map(str, (random.randint(0, 255) for _ in range(4)))))
        dstIP = socket.inet_aton(".".join(map(str, (random.randint(0, 255) for _ in range(4)))))
        protocol = random.choice([6, 17])  # TCP or UDP

        srcIP = socket.inet_aton(".".join(map(str, (random.randint(0, 255) for _ in range(4)))))
        dstIP = socket.inet_aton(".".join(map(str, (random.randint(0, 255) for _ in range(4)))))
        srcPort = struct.pack('>H', random.randint(0, 65535))
        dstPort = struct.pack('>H', random.randint(0, 65535))

        protocol = struct.pack('>B', protocol)
        tuple_data = srcIP + dstIP + srcPort + dstPort + protocol
        count = np.random.zipf(s)
        for _ in range(count):
            if i < reservoir_size:
                reservoir.append(tuple_data)
            elif i >= reservoir_size and random.random() < reservoir_size / float(i+1):
                reservoir[random.randint(0, reservoir_size-1)] = tuple_data
    with open(filename, 'wb') as f:
        for entry in reservoir:
            f.write(entry)

# Generate IPv4 data
for i in range(10):
    generate_ipv4_data('ipv4_', i, 1000, 1.5, 1000)
