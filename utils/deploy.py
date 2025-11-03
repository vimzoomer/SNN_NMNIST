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

def clogb2(depth: int) -> int:
    result = 0
    while depth > 0:
        depth >>= 1
        result += 1
    return result

def concat_layer_expr(layer_counter, neuron_cnt, spike_rd_addr, INPUT_SPIKE, NEURON, MAX_LAYER):
    SYN_G_L2 = clogb2(INPUT_SPIKE // 4 - 1)
    P = clogb2(MAX_LAYER) - clogb2(NEURON * INPUT_SPIKE // 8 - 1)
    neuron_bits = clogb2(NEURON // 2 - 1)

    neuron_cnt_masked = int(neuron_cnt) & ((1 << neuron_bits) - 1)
    spike_masked = int(spike_rd_addr) & ((1 << int(SYN_G_L2)) - 1)

    value = (
        (int(layer_counter) << (int(P) + int(neuron_bits) + int(SYN_G_L2))) |
        (neuron_cnt_masked << int(SYN_G_L2)) |
        spike_masked
    )
    return value

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    power = 1
    while power <= n:
        power *= 2
    return power

print(bin(concat_layer_expr(0, 31, 577, 2312, 64, 18496)))