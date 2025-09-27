import math
import numpy as np
import matplotlib.pyplot as plt

def c(n:int) -> int:
    if n % 2 == 0:
        return n / 2
    else:
        return 3 * n + 1

def collatz_sequence(n:int, k:int=1) -> list[int]:
    seq = [n]
    k = 0
    while seq[k] != 1:
        s = c(seq[k])
        seq.append(s)
        k += 1
        print(f"n: {n}, c(n): {seq[k]}")
    return seq

def plot_collatz_sequence(seq: list[int]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(seq, marker='o')
    plt.title("Collatz Sequence")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig("collatz_sequence.png")

def main():
    n = int(input("Enter a positive integer: "))
    sequence = collatz_sequence(n)
    print("Collatz sequence:", sequence)
    plot_collatz_sequence(sequence)

if __name__ == "__main__":
    main()