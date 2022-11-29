# ref: https://project-euphoria.dev/blog/10-lll-for-crypto/
from Crypto.Util.number import long_to_bytes

pubkey = "enter your pubkey"
ct = "enter your cipher"


def is_valid_vector(b):
    if b[0] != 0: 
        return False
    for i, x in enumerate(b):
        if i != 0 and abs(x) != 1:
            return False
    
    return True


matrix_size = len(pubkey) + 1
m_list = [
    [0 for _ in range(matrix_size) for _ in range(matrix_size)]
]

for i in range(matrix_size - 1):
    m_list[i][0] = pubkey[i]
    m_list[i][i+1] = 2
    m_list[matrix_size-1][i+1] = -1

m_list[matrix_size-1][0] = -ct

print('[+] matrix is created')

# it is only use on sagemath.
# https://www.sagemath.org/
llled = Matrix(ZZ, m_list).LLL

flag_vecs = []

for basis in llled:
    if is_valid_vector(basis):
        print('[+] found')
        flag_vecs.append(basis)

print(len(flag_vecs))

for v in flag_vecs:
    flag = ""
    for _bit in reversed(v[1:]):
        c = ("1" if _bit == 1 else "0")
        flag = c + flag

print(flag)
print(long_to_bytes(int(flag, 2)))

# %%
ascii_code = [73,84,39,83,32,83,85,80,69,82,73,78,67,82,69,65,83,73,78,71,44,32,66,65,66,89,33]
out = "text : "
for a in ascii_code:
    txt = chr(a)
    out = out + txt
print(out)
# %%
