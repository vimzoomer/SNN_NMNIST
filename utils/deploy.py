def hex_conv(num, bits):
    if num < 0:
        sign_extension = (1 << bits) - 1
        num = num & sign_extension
        hex_num = hex(num)[2:]
    else:
        binary = bin(num)[2:]
        if len(binary) < bits:
          zero_extension = "0" * int((bits - len(binary))/4)
          hex_num = zero_extension + hex(num)[2:]
        else:
          hex_num = hex(num)[2:]
    return hex_num

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    power = 1
    while power <= n:
        power *= 2
    return power