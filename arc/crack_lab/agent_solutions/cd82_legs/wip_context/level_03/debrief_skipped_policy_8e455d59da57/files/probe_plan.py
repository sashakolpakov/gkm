from collections import deque


ROWS = (
    "XXXCCCCXXX",
    "XXXCCCCXXE",
    "XXXCCCCXEE",
    "XXXXXXXEEE",
    "XXXXXXEEEE",
    "XXXXXEEEEE",
    "XXXX8EEEEE",
    "XXX88EEEEE",
    "XX888EEEEE",
    "X8888EEEEE",
)

MASKS = {
    "N": tuple(r < 5 for r in range(10) for c in range(10)),
    "NW": tuple(c <= 9 - r for r in range(10) for c in range(10)),
    "NE": tuple(c >= r for r in range(10) for c in range(10)),
    "W": tuple(c < 5 for r in range(10) for c in range(10)),
    "E": tuple(c >= 5 for r in range(10) for c in range(10)),
    "SW": tuple(c <= r for r in range(10) for c in range(10)),
    "SE": tuple(c >= 9 - r for r in range(10) for c in range(10)),
    "S": tuple(r >= 5 for r in range(10) for c in range(10)),
}


def main():
    target = tuple("XCE8".index(x) + 1 for row in ROWS for x in row)
    full = (1 << 100) - 1
    q = deque([(full, ())])
    seen = {full}
    while q:
        remaining, rev = q.popleft()
        if remaining == 0:
            print("REVERSE", rev)
            print("FORWARD", tuple(reversed(rev)))
            return
        for name, mask in MASKS.items():
            bits = [i for i, yes in enumerate(mask)
                    if yes and remaining & (1 << i)]
            if not bits:
                continue
            colors = {target[i] for i in bits}
            if len(colors) != 1:
                continue
            child = remaining
            for i, yes in enumerate(mask):
                if yes:
                    child &= ~(1 << i)
            if child not in seen:
                seen.add(child)
                q.append((child, rev + ((name, "XCE8"[colors.pop() - 1]),)))
    print("NO PLAN", len(seen))


if __name__ == "__main__":
    main()
