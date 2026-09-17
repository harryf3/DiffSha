"""Log/probability-inspired SHA-256 relaxation experiment.

This archived formulation explores an alternative arithmetic path and is not
validated as an exact SHA-256 implementation.
"""

import torch

# Max length for one block is 447 bits 
# 447+1+64 = 512

#message = torch.randint(0, 2, (447,)).to(torch.float64)

# Max length for one block is 447 bits 
# 447+1+64 = 512

#message = torch.randint(0, 2, (447,)).to(torch.float64)

def bytes_to_bits(byte_data):
    return torch.tensor([float(bit) for byte in byte_data for bit in f'{byte:08b}'])

def bits_to_bytes(tens):
    tens = tens.to(int).tolist()

    if len(tens) % 8 != 0:
        raise ValueError("Oops bits not % 8 == 0")

    byte_chunks = [
        int(''.join(map(str, tens[i:i+8])), 2)
        for i in range(0, len(tens), 8)
    ]
    return bytes(byte_chunks)

def construct_message_tensor(mess):
    """Construcsts a tensor that is the encoded message in the SHA256 format up to 512 bits
        TODO:: Extend range make work for everything should be easy enough
    
    """
    one = torch.tensor([1.0])
    message = bytes_to_bits(mess.encode("utf-8"))

    message_bits = message.shape[0]

    message_len = bytes_to_bits(message.shape[0].to_bytes(8,"big"))

    #message = torch.randint(0,2,(440,))

    message = torch.hstack((message,one,torch.zeros(447 - (message.shape[0])),message_len))
    message = message.to(torch.float64)
    return message, message_bits
    # message.requires_grad = True

    # og_message = message.clone().detach()

coerce_steepness = 6.5
coerce_offset = 3.5
# this seems like a relatively safe number but have to test with multiple hashes 
# Could possibly be a little smaller

def coerce(tens):
    #return torch.exp(coerce_steepness*tens - coerce_offset)/(1+torch.exp(coerce_steepness*tens - coerce_offset))
    return tens

def right_rotate(tens, by):
    # Doesn't need to be coerced
    return torch.roll(tens,by)

def bitnot(tens):
    #return (tens-1)**2 # maybe problem since assumes p*p so now chance is less likely where as abs would make chance
    # Doesn't need to be coerced
    #return torch.abs(tens-1.0)
    return torch.abs(1.0 - tens)
    #return 5.30817622906 * torch.log(torch.cosh(tens-1.0)) # 5.3081 ... is the multiplier that lets bitnot(0) = 1 instead of 1/5.30....

def bitand(a,b):
    # This offers a alternative so we don't have so many mults of vals in backward pass 
    # maybe even torch.log((a+b+2 - torch.abs(a-b))/2.0)
    #return a*b
    return a+b
    #return a*b

def log_xor(tens1,tens2):
    return torch.log(tens1 + tens2) / (2*(torch.log(tens1) +torch.log(tens2)))

def bitwise_add(a, b):
    # This is probably what is causing big big grads and slowness 
    # Kind of stuck with it I suppose you could go to number than back 
    for _ in range(32):
        carry = left_shift(bitand(a,b),1)
        a = xor(a,b)
        b = carry
    return torch.exp(a)

def bitwise_adds(lis):
    # Doesn't need to be coerced since bitwise_add is 
    running  = lis[0]
    for i in range(len(lis)):
        if(i>=len(lis)-1):
            return running
        running = bitwise_add(running,lis[i+1])
    return running

def right_shift(tens, by):
    # No need to coerce
    if(by >= len(tens)):
        by = len(tens)
    return torch.cat((torch.zeros(by), tens[:-by]))

def left_shift(tens,by):
    # No need to coerce
    if(by >= len(tens)):
        by = len(tens)
    return torch.cat((tens[by:],torch.zeros(by)))

def xor(tens1,tens2):
    # Must be coerced 
    # When you do a ton of xors like in the bitwise add it goes to 0.5
    return coerce((tens1 + tens2) - 2*bitand(tens1,tens2))

def sigma0(tens):
    # Interesting to note bit probs post op are ~ gaussian
    return xor(xor(right_rotate(tens, 7), right_rotate(tens, 18)), right_shift(tens,3)) 

def sigma1(tens):
    return xor(xor(right_rotate(tens, 17), right_rotate(tens, 19)), right_shift(tens,10))

def capsigma0(tens):
    return xor(xor(right_rotate(tens, 2),right_rotate(tens, 13)),right_rotate(tens, 22))

def capsigma1(tens):
    return xor(xor(right_rotate(tens, 6),right_rotate(tens, 11)),right_rotate(tens, 25))

def ch(tens1, tens2, tens3):
    return xor(bitand(tens1,tens2), bitand(bitnot(tens1),tens3))

def maj(tens1, tens2, tens3):
    return xor(xor( bitand(tens1,tens2), bitand(tens1,tens3)), bitand(tens2,tens3))

class LogProbabilitySHA256:
    def __init__(self):
        self.K = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]

        self.tK = []

        for initial_num in self.K:
            self.tK.append(bytes_to_bits(initial_num.to_bytes(4,"big")))

        self.h0 = 0x6a09e667
        self.h1 = 0xbb67ae85
        self.h2 = 0x3c6ef372
        self.h3 = 0xa54ff53a
        self.h5 = 0x9b05688c
        self.h4 = 0x510e527f
        self.h6 = 0x1f83d9ab
        self.h7 = 0x5be0cd19

        self.th0 = bytes_to_bits(self.h0.to_bytes(4,"big"))
        self.th1 = bytes_to_bits(self.h1.to_bytes(4,"big"))
        self.th2 = bytes_to_bits(self.h2.to_bytes(4,"big"))
        self.th3 = bytes_to_bits(self.h3.to_bytes(4,"big"))
        self.th4 = bytes_to_bits(self.h4.to_bytes(4,"big"))
        self.th5 = bytes_to_bits(self.h5.to_bytes(4,"big"))
        self.th6 = bytes_to_bits(self.h6.to_bytes(4,"big"))
        self.th7 = bytes_to_bits(self.h7.to_bytes(4,"big"))

    def do_hash(self,message,rounds=64):
        #print(message)
        message_schedule = []

        th0 = bytes_to_bits(self.h0.to_bytes(4,"big"))
        th1 = bytes_to_bits(self.h1.to_bytes(4,"big"))
        th2 = bytes_to_bits(self.h2.to_bytes(4,"big"))
        th3 = bytes_to_bits(self.h3.to_bytes(4,"big"))
        th4 = bytes_to_bits(self.h4.to_bytes(4,"big"))
        th5 = bytes_to_bits(self.h5.to_bytes(4,"big"))
        th6 = bytes_to_bits(self.h6.to_bytes(4,"big"))
        th7 = bytes_to_bits(self.h7.to_bytes(4,"big"))
        
        for t in range(0, 64):
            if t <= 15:
                # adds the t'th 32 bit word of the block,
                # starting from leftmost word
                # 4 bytes at a time
                message_schedule.append(message[t*32:(t*32)+32])
            else:
                term1 = sigma1(message_schedule[t-2])
                term2 = message_schedule[t-7]
                term3 = sigma0(message_schedule[t-15])
                term4 = message_schedule[t-16]

                # print(f"Round {t}")
                # print(f"T1 {term1}")
                # print(f"T2 {term2}")
                # print(f"T3 {term3}")
                # print(f"T4 {term4}")

                # # append a 4-byte byte object
                schedule = bitwise_adds([term1, term2, term3, term4])# We work in bits no need to modulo 32 
                # print(f"Schedule {schedule}")
                # print("#"*50)
                message_schedule.append(schedule)

        #print(message_schedule)

        a = th0
        b = th1
        c = th2
        d = th3
        e = th4
        f = th5
        g = th6
        h = th7

        # Iterate for t=0 to 63
        for t in range(rounds):

            t1 = bitwise_adds([h, capsigma1(e), ch(e, f, g), self.tK[t], message_schedule[t]])
            t2 = bitwise_adds([capsigma0(a), maj(a, b, c)])
            h = g
            g = f
            f = e
            #print(f"e : {bits_to_bytes(bitwise_add(d,t1))}")
            e = bitwise_add(d,t1) # No need to modulo we are constrained to 32 bits
            d = c
            c = b
            b = a
            #print(f"a : {bits_to_bytes(bitwise_add(t1,t2))}")
            a = bitwise_add(t1,t2)

            # print(f"{t} " +"#"*25)
            # gpl = torch.sum(t1.clone())
            # gpl.backward()
            # print(message.grad)

            # print(f"{t} " + "#"*25)
            # print(f"T1 : {bits_to_bytes(t1)}")
            # print(f"T2 : {bits_to_bytes(t2)}")
            # print(f"working : {[bits_to_bytes(a),bits_to_bytes(b),bits_to_bytes(c),bits_to_bytes(d),bits_to_bytes(e),bits_to_bytes(f),bits_to_bytes(g),bits_to_bytes(h)]}")
            # print(f"T1 : {t1}")
            # print(f"T2 : {t2}")
            # print(f"working : {[a,b,c,d,e,f,g,h]}")

        th0 = bitwise_add(th0,a) 
        th1 = bitwise_add(th1,b)
        th2 = bitwise_add(th2,c) 
        th3 = bitwise_add(th3,d)
        th4 = bitwise_add(th4,e)
        th5 = bitwise_add(th5,f)
        th6 = bitwise_add(th6,g) 
        th7 = bitwise_add(th7,h) 

        out = torch.concat((th0,th1,th2,th3,th4,th5,th6,th7))
        return out
        #out = F.sigmoid(out-1.0)
