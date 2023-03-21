from minitorch import Scalar, tensor, Tensor, grad_check

r_order = [0, 4, 3, 1, 2]
r_order = [a[0] for a in sorted(enumerate(r_order), key=lambda a: a[1])]
t = tensor([0])
# here 0 means to reduce the 0th dim, 3 -> nothing
permutation = [0]


def permute(a: Tensor) -> Tensor:
    return a.permute(*permutation)


grad_check(permute, t)
a = Scalar(1.0)
b = Scalar(2.0)
c = (a + b) * b
print(c)
