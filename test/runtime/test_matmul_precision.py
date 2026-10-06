import unittest
from tinygrad import Context, Tensor, dtypes


class TestMatmulPrecision(unittest.TestCase):
  def test_fallback_products(self):
    cases = ((dtypes.fp8e4m3, 96.0), (dtypes.fp8e5m2, 1.25), (dtypes.half, 256.0),
             (dtypes.half, 1+2**-10), (dtypes.bfloat16, 1+2**-7))
    for dt, value in cases:
      for size, tc in ((1, 1), (32, 0)):
        with self.subTest(dtype=dt, size=size, tc=tc), Context(TC=tc):
          a = Tensor([[value]*size for _ in range(size)], dtype=dt).realize()
          result = a.matmul(a, dtype=dtypes.float32)
          self.assertEqual(result.tolist(), [[size*value*value]*size for _ in range(size)])


if __name__ == "__main__": unittest.main()
