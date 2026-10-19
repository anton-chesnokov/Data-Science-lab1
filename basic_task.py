from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if not predicted or len(predicted) != len(expected):
        raise ValueError("Vectors must have the same non-zero length.")
    squared_errors = [(p - e) ** 2 for p, e in zip(predicted, expected)]
    return sum(squared_errors) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                count += 1
                n //= d
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    
    result = []
    for prime, power in sorted(factors):
        if power == 1:
            result.append(f"({prime})")
        else:
            result.append(f"({prime}**{power})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    n = data.value
    k = 1
    current_sum = 0
    while current_sum < n:
        current_sum += k * k
        if current_sum == n:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    length = len(s)
    mid = length // 2
    if length % 2 == 0:
        left_part = s[:mid]
        right_part = s[mid:]
    else:
        left_part = s[:mid]
        right_part = s[mid + 1:]
    
    sum_left = sum(int(digit) for digit in left_part)
    sum_right = sum(int(digit) for digit in right_part)
    return sum_left == sum_right