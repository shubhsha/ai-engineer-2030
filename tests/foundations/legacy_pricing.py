# ruff: noqa
# The pricing function as it was found: kept verbatim, never edited, so the
# characterization test can compare old and new behavior.


def calc(items, cust):
    t = 0
    for i in items:
        p = i[1] * i[2]
        if i[2] >= 10:
            p = p - p * 0.05
        t = t + p
    if cust.get("loyal") == True:
        t = t - t * 0.03
    t = t + t * 0.08
    return round(t, 2)
