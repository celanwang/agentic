"""Frozen arrival-order, first-fit packer. Sizes and capacity share one unit."""
def pack(capacity, count, sizes):
    bins = [[] for _ in range(count)]
    remaining = [capacity] * count
    for i, size in enumerate(sizes):
        for j, free in enumerate(remaining):
            if size <= free:
                bins[j].append(i)
                remaining[j] -= size
                break
        else:
            return None
    return bins

if __name__ == "__main__":
    print(pack(10, 2, [4, 4, 6, 6]))
